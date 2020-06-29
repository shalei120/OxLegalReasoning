import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter

import numpy as np

import datetime


from Encoder import Encoder
from Decoder import Decoder
from Hyperparameters import args

class LSTM_iterIB_Model(nn.Module):
    """
    Implementation of a seq2seq model.
    Architecture:
        Encoder/decoder
        2 LTSM layers
    """

    def __init__(self, w2i, i2w):
        """
        Args:
            args: parameters of the model
            textData: the dataset object
        """
        super(LSTM_iterIB_Model, self).__init__()
        print("Model creation...")

        self.word2index = w2i
        self.index2word = i2w
        self.max_length = args['maxLengthDeco']

        self.NLLloss = torch.nn.NLLLoss(reduction = 'none')
        self.CEloss =  torch.nn.CrossEntropyLoss(reduction = 'none')

        self.embedding = nn.Embedding(args['vocabularySize'], args['embeddingSize']).to(args['device'])

        self.encoder = Encoder(w2i, i2w, self.embedding).to(args['device'])

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = -1)

        self.z_init2nero_mu = nn.Linear(args['hiddenSize'], args['hiddenSize']).to(args['device'])
        self.z_init2nero_logvar = nn.Linear(args['hiddenSize'], args['hiddenSize']).to(args['device'])
        self.z_sym_to_qz_mu= nn.Linear(args['hiddenSize'], args['hiddenSize']).to(args['device'])
        self.z_sym_to_qz_logvar = nn.Linear(args['hiddenSize'], args['hiddenSize']).to(args['device'])

        self.x_2_prob_z = nn.Sequential(
            nn.Linear(args['hiddenSize'], 2)
          ).to(args['device'])
        # self.z_to_fea = nn.Sequential(
        #     nn.Linear(args['hiddenSize'], args['hiddenSize']).to(args['device']),
        #     nn.Tanh()
        #   ).to(args['device'])


        self.ChargeClassifier = nn.Sequential(
            nn.Linear(args['hiddenSize'], args['chargenum']+2),   # add EOS charge
            nn.LogSoftmax(dim=-1)
          ).to(args['device'])

        self.EOS_charge_index = args['chargenum']
        self.PAD_charge_index = args['chargenum'] + 1
        self.max_label = 5
        
    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).to(args['device'])
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature = args['temperature']):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y
        return y_hard

    def sample_z(self, mu, log_var):
        eps = Variable(torch.randn(mu.size())).to(args['device'])
        return mu + torch.exp(log_var / 2) * eps

    def forward(self, x, eps = 0.000001):
        '''
        :param encoderInputs: [batch, enc_len]
        :param decoderInputs: [batch, dec_len]
        :param decoderTargets: [batch, dec_len]
        :return:
        '''

        # print(x['enc_input'])
        self.encoderInputs = x['enc_input'].to(args['device'])
        self.encoder_lengths = x['enc_len']
        self.classifyLabels = x['labels'].to(args['device'])
        self.batch_size = self.encoderInputs.size()[0]
        self.seqlen = self.encoderInputs.size()[1]

        label_len = self.classifyLabels.size()[1]

        mask = torch.sign(self.encoderInputs).float()
        mask_label = (self.classifyLabels != self.PAD_charge_index).float()

        en_outputs, en_state = self.encoder(self.encoderInputs, self.encoder_lengths)  # batch seq hid

        loss = 0

        for ind in range(label_len):
            x0 = en_outputs * mask.unsqueeze(2)
            z_nero_init, _ = torch.max(en_outputs, dim = 1)
            z_nero_mu = self.z_init2nero_mu(z_nero_init)
            z_nero_logvar = self.z_init2nero_logvar(z_nero_init)

            I_x_z_nero = torch.sum(0.5 * (torch.exp(z_nero_logvar) + z_nero_mu ** 2  - 1 - z_nero_logvar))

            z_logits = self.x_2_prob_z(x0.to(args['device'])) # batch seq 2
            z_logits_fla = z_logits.reshape((self.batch_size * self.seqlen, 2))

            z_prob = self.softmax(z_logits)
            I_x_z_sym = torch.mean(torch.sum(-torch.log(z_prob[:,:,0]+ eps), dim = 1))

            sampled_seq = self.gumbel_softmax(z_logits_fla).reshape((self.batch_size, self.seqlen, 2))  # batch seq  //0-1
            sampled_seq = sampled_seq * mask.unsqueeze(2)

            sampled_num = torch.sum(sampled_seq[:,:,1], dim = 1) # batch
            sampled_num = (sampled_num == 0).to(args['device'], dtype=torch.float32)  + sampled_num
            z_sym = x0 * (sampled_seq[:,:,1].unsqueeze(2))  # batch seq hid

            ''' z_sym -> z_nero'''
            z_sym_mean = torch.sum(z_sym, dim = 1)/ sampled_num.unsqueeze(1)# batch hid
            qz_mu = self.z_sym_to_qz_mu(z_sym_mean)
            qz_logvar = self.z_sym_to_qz_logvar(z_sym_mean)
            z2_sampled = self.sample_z(z_nero_mu, z_nero_logvar)

            # print(z2_sampled.size(), qz_mu.size(), qz_logvar.size())
            I_z1_z2 = -0.5* (z2_sampled - qz_mu)**2/(torch.exp(qz_logvar)) -0.5*qz_logvar # batch hid
            I_z1_z2 = torch.mean(I_z1_z2)

            z2_sampled_for_cla = self.sample_z(z_nero_mu, z_nero_logvar)


            output = self.ChargeClassifier(z2_sampled_for_cla).to(args['device'])  # batch chargenum

            recon_loss = self.NLLloss(output, self.classifyLabels[:,ind]).to(args['device'])
            recon_loss = recon_loss * mask_label[:,ind]
            recon_loss_mean = torch.mean(recon_loss).to(args['device'])

            loss = loss + recon_loss_mean + 0.01* I_x_z_sym - I_z1_z2 +I_x_z_nero

            # mask = mask * sampled_seq[:,:,1]


        return loss

    def predict(self, x):
        encoderInputs = x['enc_input'].to(args['device'])
        encoder_lengths = x['enc_len']

        batch_size = encoderInputs.size()[0]
        seqlen = encoderInputs.size()[1]

        # label_len = self.classifyLabels.size()[1]

        mask = torch.sign(encoderInputs).float()


        en_outputs, en_state = self.encoder(encoderInputs, encoder_lengths)  # batch seq hid


        answer = []

        for ind in range(self.max_label):
            x0 = en_outputs * mask.unsqueeze(2)
            z_nero_init, _ = torch.max(en_outputs, dim=1)
            z_nero_mu = self.z_init2nero_mu(z_nero_init)
            z_nero_logvar = self.z_init2nero_logvar(z_nero_init)

            z_logits = self.x_2_prob_z(x0.to(args['device']))  # batch seq 2
            z_logits_fla = z_logits.reshape((batch_size * seqlen, 2))


            sampled_seq = self.gumbel_softmax(z_logits_fla).reshape(
                (batch_size, seqlen, 2))  # batch seq  //0-1
            sampled_seq = sampled_seq * mask.unsqueeze(2)

            sampled_num = torch.sum(sampled_seq[:, :, 1], dim=1)  # batch
            sampled_num = (sampled_num == 0).to(args['device'], dtype=torch.float32) + sampled_num
            z_sym = x0 * (sampled_seq[:, :, 1].unsqueeze(2))  # batch seq hid

            ''' z_sym -> z_nero'''
            z_sym_mean = torch.sum(z_sym, dim=1) / sampled_num.unsqueeze(1)  # batch hid
            qz_mu = self.z_sym_to_qz_mu(z_sym_mean)
            qz_logvar = self.z_sym_to_qz_logvar(z_sym)
            z2_sampled = self.sample_z(z_nero_mu, z_nero_logvar)

            z2_sampled_for_cla = self.sample_z(z_nero_mu, z_nero_logvar)

            output = self.ChargeClassifier(z2_sampled_for_cla).to(args['device'])  # batch chargenum
            ans = torch.argmax(output, dim = 1)
            answer.append(ans)
            # mask = mask * sampled_seq[:,:,1]

        answer = torch.stack(answer)
        finalanswer = []
        for b in range(batch_size):
            decode_id_list = list(answer[:,b])
            if self.EOS_charge_index in decode_id_list:
                decode_id_list = decode_id_list[:decode_id_list.index(self.EOS_charge_index)] \
                    if decode_id_list[0] != self.EOS_charge_index else \
                    [self.EOS_charge_index]
            finalanswer.append(decode_id_list)



        return finalanswer
