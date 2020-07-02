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

class LSTM_GMIB_Model(nn.Module):
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
        super(LSTM_GMIB_Model, self).__init__()
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
        self.sigmoid = nn.Sigmoid()

        self.x_2_prob_z = nn.Sequential(
            nn.Linear(args['hiddenSize'], 2)
          ).to(args['device'])
        self.z_to_fea = nn.Linear(args['hiddenSize'], args['hiddenSize']).to(args['device'])

        self.ChargeClassifier = nn.Sequential(
            nn.Linear(args['hiddenSize'], args['chargenum']),
            nn.LogSoftmax(dim=-1)
          ).to(args['device'])

        self.charge_dist_mu = Parameter(torch.rand(args['chargenum'] + 1, args['hiddenSize'])).to(args['device'])
        self.charge_dist_logvar = Parameter(torch.rand(args['chargenum'] + 1, args['hiddenSize'])).to(args['device'])

        self.P_z_w_mu = nn.Sequential(
            nn.Linear(args['hiddenSize'], args['hiddenSize']),
            nn.Tanh()
          ).to(args['device'])
        self.P_z_w_logvar = nn.Sequential(
            nn.Linear(args['hiddenSize'], args['hiddenSize']),
            nn.Tanh()
        ).to(args['device'])

        self.dec_P_x_z = nn.Sequential(
            nn.Linear(args['hiddenSize'], args['embeddingSize']),
            nn.Tanh()
          ).to(args['device'])

        
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

        mask = torch.sign(self.encoderInputs).float()

        en_outputs, en_state = self.encoder(self.encoderInputs, self.encoder_lengths)  # batch seq hid

        # print(en_outputs.size())
        # z_logits = self.x_2_prob_z(en_outputs.to(args['device'])) # batch seq 2
        #
        #
        # z_logits_fla = z_logits.reshape((self.batch_size * self.seqlen, 2))
        # sampled_seq = self.gumbel_softmax(z_logits_fla).reshape((self.batch_size, self.seqlen, 2))  # batch seq  //0-1
        # sampled_seq = sampled_seq * mask.unsqueeze(2)
        #
        # # print(sampled_seq)
        #
        # sampled_num = torch.sum(sampled_seq[:,:,1], dim = 1) # batch
        # sampled_num = (sampled_num == 0).to(args['device'], dtype=torch.float32)  + sampled_num
        # sampled_word = en_outputs * (sampled_seq[:,:,1].unsqueeze(2))  # batch seq hid
        # # print(sampled_word[0,:,:])
        # # exit()
        s_w_feature = self.z_to_fea(en_outputs)
        s_w_feature,_ = torch.max(s_w_feature, dim = 1)# batch hid

        words_mu = self.P_z_w_mu(en_outputs)    # batch seq hid # 1
        words_logvar = self.P_z_w_logvar(en_outputs)

        words_z = self.sample_z(words_mu, words_logvar)  # batch seq hid

        logit_P_c_w = -0.5*((words_z ** 2)@ torch.exp(-self.charge_dist_logvar).transpose(0,1)
                                       -2*words_z @ (self.charge_dist_mu/torch.exp(self.charge_dist_logvar)).transpose(0,1)
                                       + torch.sum(self.charge_dist_mu**2/torch.exp(self.charge_dist_logvar), dim = 1).unsqueeze(0).unsqueeze(1) )\
                     - torch.sum(0.5 * self.charge_dist_logvar, dim = -1).unsqueeze(0).unsqueeze(1)  # batch seq charge

        words_c = self.gumbel_softmax(logit_P_c_w)  # batch seq charge

        words_cz_mu = words_c @ self.charge_dist_mu # batch seq hid #0
        words_cz_logvar = words_c @ self.charge_dist_logvar # batch seq hid

        KL_cz_z = 0.5* (torch.exp(words_cz_logvar - words_logvar) + (words_mu - words_cz_mu) ** 2 / torch.exp(words_logvar) - 1 + words_logvar - words_cz_logvar)
        KL_cz_z = torch.sum(KL_cz_z, dim = 2) * mask # batch seq
        KL_cz_z = torch.mean(KL_cz_z)

        w_prime = self.dec_P_x_z(words_z) # b s e
        P_recon = self.softmax(w_prime @ self.embedding.weight.transpose(0,1)) # b s v
        recon_loss = self.CEloss(P_recon.transpose(1,2), self.encoderInputs) * mask
        recon_loss_mean = torch.mean(recon_loss).to(args['device'])

        y = F.one_hot(self.classifyLabels, num_classes=args['chargenum'] + 2)
        y = y[:, :, :(args['chargenum']+1)]  # add content class
        y = torch.sum(y, dim=1)  # b chargenum
        P_c = (y.float() + 0.00001) / torch.sum(y.float() + 0.00001, dim = 1, keepdim=True)
        P_c_w = logit_P_c_w / torch.sum(logit_P_c_w, dim = -1, keepdim=True) # batch seq charge
        KL_c = torch.sum(P_c_w * torch.log(P_c_w / P_c.unsqueeze(1)), dim = 2)
        KL_c = torch.mean(KL_c)

        KL_origin = torch.mean(torch.sum(0.5 * (torch.exp(self.charge_dist_logvar) + self.charge_dist_mu ** 2  - 1 - self.charge_dist_logvar), dim = 1))

        sum_P_c_w = torch.sum(P_c_w, dim = 1) # batch charge
        P_stat = sum_P_c_w / torch.sum(sum_P_c_w, dim = 1, keepdim=True)


        I_x_z = torch.mean(torch.sum(-torch.log(P_c_w[:,:,args['chargenum']]+ eps), dim = 1))
        # print(I_x_z)
        # en_hidden, en_cell = en_state   #2 batch hid
        # print(z_prob[0,:,:], sampled_num, I_x_z, torch.sum(-torch.log(z_prob[0,:,0]+ eps)))

        pred = self.sigmoid(s_w_feature @ self.charge_dist_mu.transpose(0, 1)).to(args['device'])  # batch c
        pred_p = (pred + P_stat) / 2
        cla_loss = y.float() * torch.log(pred_p) + (1 - y.float()) * torch.log(1 - pred_p)
        cla_loss_mean = -torch.mean(torch.sum(cla_loss, dim=1))

        loss = cla_loss_mean + recon_loss_mean + KL_c + KL_cz_z + KL_origin + 0.02 * I_x_z

        # wordnum = torch.sum(mask, dim = 1)
        # print(sampled_num / wordnum)
        # exit()

        return loss

    def predict(self, x):
        encoderInputs = x['enc_input'].to(args['device'])
        encoder_lengths = x['enc_len']

        batch_size = encoderInputs.size()[0]
        seqlen = encoderInputs.size()[1]
        mask = torch.sign(encoderInputs).float()

        en_outputs, en_state = self.encoder(encoderInputs, encoder_lengths)

        s_w_feature = self.z_to_fea(en_outputs)
        s_w_feature, _ = torch.max(s_w_feature, dim=1)  # batch hid

        words_mu = self.P_z_w_mu(en_outputs)  # batch seq hid # 1
        words_logvar = self.P_z_w_logvar(en_outputs)

        words_z = self.sample_z(words_mu, words_logvar)  # batch seq hid


        logit_P_c_w = -0.5*((words_z ** 2)@ torch.exp(-self.charge_dist_logvar).transpose(0,1)
                                       -2*words_z @ (self.charge_dist_mu/torch.exp(self.charge_dist_logvar)).transpose(0,1)
                                       + torch.sum(self.charge_dist_mu**2/torch.exp(self.charge_dist_logvar), dim = 1).unsqueeze(0).unsqueeze(1) )\
                     - torch.sum(0.5 * self.charge_dist_logvar, dim = -1).unsqueeze(0).unsqueeze(1)  # batch seq charge

        words_c = self.gumbel_softmax(logit_P_c_w)  # batch seq charge

        words_cz_mu = words_c @ self.charge_dist_mu  # batch seq hid #0
        words_cz_logvar = words_c @ self.charge_dist_logvar  # batch seq hid

        # KL_cz_z = 0.5 * (torch.exp(words_cz_logvar - words_logvar) + (words_mu - words_cz_mu) ** 2 / torch.exp(
        #     words_logvar) - 1 + words_logvar - words_cz_logvar)
        # KL_cz_z = torch.sum(KL_cz_z, dim=2) * mask  # batch seq
        # KL_cz_z = torch.mean(KL_cz_z)

        w_prime = self.dec_P_x_z(words_z)  # b s e
        P_recon = self.softmax(w_prime @ self.embedding.weight.transpose(0, 1))  # b s v
        # recon_loss = self.CEloss(P_recon, self.encoderInputs) * mask
        # recon_loss_mean = torch.mean(recon_loss).to(args['device'])

        y = F.one_hot(self.classifyLabels, num_classes=args['chargenum'] + 2)
        y = y[:, :, :(args['chargenum'] + 1)]  # add content class
        y = torch.sum(y, dim=1)  # b chargenum
        P_c = (y.float() + 0.00001) / torch.sum(y.float() + 0.00001, dim = 1, keepdim=True)
        P_c_w = logit_P_c_w / torch.sum(logit_P_c_w, dim=-1, keepdim=True)  # batch seq charge
        # KL_c = torch.sum(P_c_w * torch.log(P_c_w / P_c.unsqueeze(1)), dim=2)
        # KL_c = torch.mean(KL_c)

        # KL_origin = torch.mean(torch.sum(
        #     0.5 * (torch.exp(self.charge_dist_logvar) + self.charge_dist_mu ** 2 - 1 - self.charge_dist_logvar), dim=1))

        sum_P_c_w = torch.sum(P_c_w, dim=1)  # batch charge
        P_stat = sum_P_c_w / torch.sum(sum_P_c_w, dim=1, keepdim=True)

        # I_x_z = torch.mean(torch.sum(-torch.log(P_c_w[:, :, args['chargenum']] + eps), dim=1))
        # print(I_x_z)
        # en_hidden, en_cell = en_state   #2 batch hid
        # print(z_prob[0,:,:], sampled_num, I_x_z, torch.sum(-torch.log(z_prob[0,:,0]+ eps)))

        pred = self.sigmoid(s_w_feature @ self.charge_dist_mu.transpose(0, 1)).to(args['device'])  # batch c
        pred_p = (pred + P_stat) / 2
        # cla_loss = y.float() * torch.log(pred_p) + (1 - y.float()) * torch.log(1 - pred_p)
        # cla_loss_mean = -torch.mean(torch.sum(cla_loss, dim=1))

        choose_res = torch.argmax(pred_p, dim = -1)
        wordnum = torch.sum(mask, dim = 1)

        finalanswer = []

        for b in range(batch_size):
            decode_id_list = []
            for ind, choose in enumerate(choose_res[b, :]):
                if choose == 1:
                    decode_id_list.append(ind)
            if len(decode_id_list) == 0:
                decode_id_list.append(torch.argmax(pred_p[b, :, 1]))
            finalanswer.append(decode_id_list)

        return finalanswer
