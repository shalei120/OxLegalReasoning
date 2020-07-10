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
        self.additive = True

        self.word2index = w2i
        self.index2word = i2w
        self.max_length = args['maxLengthDeco']

        self.NLLloss = torch.nn.NLLLoss(reduction = 'none')
        self.CEloss =  torch.nn.CrossEntropyLoss(reduction = 'none')

        self.embedding = nn.Embedding(args['vocabularySize'], args['embeddingSize'])

        self.encoder = Encoder(w2i, i2w, self.embedding)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = -1)
        self.sigmoid = nn.Sigmoid()

        # self.x_2_prob_z = nn.Sequential(
        #     nn.Linear(args['hiddenSize'], 2)
        #   )
        self.z_to_fea = nn.Linear(args['hiddenSize']*2, args['hiddenSize'])

        self.ChargeClassifier = nn.Sequential(
            nn.Linear(args['hiddenSize']*2, args['chargenum']),
            nn.Sigmoid()
          )

        self.charge_dist_mu = Parameter(torch.rand(args['chargenum'] + 1, args['hiddenSize']))
        self.charge_dist_logvar = Parameter(torch.rand(args['chargenum'] + 1, args['hiddenSize']))

        self.P_z_w_mu = nn.Sequential(
            nn.Linear(args['embeddingSize'], args['hiddenSize']),
            nn.Tanh()
          )
        self.P_z_w_logvar = nn.Sequential(
            nn.Linear(args['embeddingSize'], args['hiddenSize']),
            nn.Tanh()
        )

        self.dec_P_x_z = nn.Sequential(
            nn.Linear(args['hiddenSize'], args['embeddingSize']),
            nn.Tanh()
          )

        
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

    def build(self, x, eps = 1e-6):
        '''
        :param encoderInputs: [batch, enc_len]
        :param decoderInputs: [batch, dec_len]
        :param decoderTargets: [batch, dec_len]
        :return:
        '''

        # print(x['enc_input'])
        self.encoderInputs = x['enc_input']
        self.encoder_lengths = x['enc_len']
        self.classifyLabels = x['labels']
        self.batch_size = self.encoderInputs.size()[0]
        self.seqlen = self.encoderInputs.size()[1]

        mask = torch.sign(self.encoderInputs).float()
        enc_input_embeddings = self.embedding(self.encoderInputs)

        en_outputs, en_state = self.encoder(self.encoderInputs, self.encoder_lengths)  # batch seq hid

        # print(en_outputs.size())
        # z_logits = self.x_2_prob_z(en_outputs) # batch seq 2
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

        # GM
        words_mu = self.P_z_w_mu(enc_input_embeddings)    # batch seq hid # 1
        words_logvar = self.P_z_w_logvar(enc_input_embeddings)

        words_z = self.sample_z(words_mu, words_logvar)  # batch seq hid

        # logit_P_c_w = -0.5*((words_z ** 2)@ torch.exp(-self.charge_dist_logvar).transpose(0,1)
        #                                -2*words_z @ (self.charge_dist_mu/torch.exp(self.charge_dist_logvar)).transpose(0,1)
        #                                + torch.sum(self.charge_dist_mu**2/torch.exp(self.charge_dist_logvar), dim = 1).unsqueeze(0).unsqueeze(1) )\
        #              - torch.sum(0.5 * self.charge_dist_logvar, dim = -1).unsqueeze(0).unsqueeze(1)  # batch seq charge
        logit_P_c_w = torch.einsum('bsh,ch->bsc', words_z, self.charge_dist_mu)


        y = F.one_hot(self.classifyLabels, num_classes=args['chargenum'] + 2)
        y = y[:, :, :(args['chargenum'] + 1)]  # add content class
        y, _ = torch.max(y, dim=1)  # b chargenum+1
        logit_P_c_w_mask = logit_P_c_w * y.unsqueeze(1).float() + (1- y.unsqueeze(1).float()) * (-1e20)

        words_c = self.gumbel_softmax(logit_P_c_w_mask)  # batch seq charge

        words_cz_mu = words_c @ self.charge_dist_mu # batch seq hid #0
        words_cz_logvar = words_c @ self.charge_dist_logvar # batch seq hid

        KL_cz_z = 0.5* (torch.exp(words_cz_logvar - words_logvar) + (words_mu - words_cz_mu) ** 2 / torch.exp(words_logvar) - 1 + words_logvar - words_cz_logvar)
        KL_cz_z = torch.sum(KL_cz_z, dim = 2) * mask # batch seq
        KL_cz_z = torch.mean(KL_cz_z)

        w_prime = self.dec_P_x_z(words_z) # b s e
        P_recon = w_prime @ self.embedding.weight.transpose(0,1) # b s v
        recon_loss = self.CEloss(P_recon.transpose(1,2), self.encoderInputs) * mask
        recon_loss_mean = torch.mean(recon_loss)


        P_c = (y.float() + 0.00001) / torch.sum(y.float() + 0.00001, dim = 1, keepdim=True)
        P_c_w = self.softmax(logit_P_c_w)  # batch seq charge
        KL_c = torch.sum(P_c_w * torch.log(P_c_w / P_c.unsqueeze(1) + eps), dim = 2)
        KL_c = torch.mean(KL_c)
        H_c_w = -torch.sum(P_c_w * torch.log(P_c_w + eps), dim = 2)
        H_c_w = torch.mean(H_c_w)

        KL_origin = torch.mean(0.5 * (torch.exp(self.charge_dist_logvar) + self.charge_dist_mu ** 2  - 1 - self.charge_dist_logvar))

        sum_P_c_w = torch.sum(P_c_w, dim = 1)[:,:args['chargenum']] # batch charge
        # P_stat = sum_P_c_w / torch.sum(sum_P_c_w, dim = 1, keepdim=True)
        sum_P_c_w_max,_ = torch.max(sum_P_c_w, dim = 1, keepdim=True)
        P_stat = sum_P_c_w / sum_P_c_w_max * ((sum_P_c_w_max+eps) / (1+sum_P_c_w_max + eps))

        chargefea = self.tanh(P_stat @ self.charge_dist_mu[:args['chargenum'],:])

        I_x_z = torch.mean(-torch.log(P_c_w[:,:,args['chargenum']]+ eps))

        if self.additive:
            # pred = self.sigmoid(s_w_feature @ self.charge_dist_mu[:args['chargenum'],:].transpose(0, 1))  # batch c
            pred = self.ChargeClassifier(torch.cat([s_w_feature, chargefea], dim = 1) )
            pred_p = pred #(pred + P_stat) / 2
        else:
            pred_p = self.ChargeClassifier(torch.cat([s_w_feature, P_stat], dim = 1))

        cla_loss = y[:,:args['chargenum']].float() * torch.log(pred + eps) + (1 - y[:,:args['chargenum']].float()) * torch.log(1 - pred + eps)
        cla_loss_mean = -torch.mean(torch.sum(cla_loss, dim=1))

        # cla_loss_GM = y[:,:args['chargenum']].float() * torch.log(P_stat) + (1 - y[:,:args['chargenum']].float()) * torch.log(1 - P_stat)
        cla_loss_GM_mean = 0#-torch.mean(torch.sum(cla_loss_GM, dim=1))

        loss = cla_loss_mean  + recon_loss_mean + KL_c + KL_cz_z + KL_origin + 0.02 * I_x_z + H_c_w
        tt =  torch.stack([cla_loss_mean, recon_loss_mean, KL_c, KL_cz_z,  KL_origin,  I_x_z, H_c_w])
        if any(tt != tt):
            print(tt)
            exit()
        return loss, tt, pred_p

    def forward(self, x):
        loss, tt, _ = self.build(x)

        return loss, tt

    def predict(self, x):
        _,_, pred_p = self.build(x)
        choose_res = (pred_p > 0.5)
        max_choose, _ = torch.max(pred_p, dim = 1)
        choose_res = choose_res | (pred_p == max_choose.unsqueeze(1))
        # wordnum = torch.sum(mask, dim = 1)


        return choose_res
