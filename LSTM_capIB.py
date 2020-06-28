import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter

import numpy as np

import datetime
import math

from Encoder import Encoder
from Decoder import Decoder
from Hyperparameters import args

class LSTM_capsule_IB_Model(nn.Module):
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
        super(LSTM_capsule_IB_Model, self).__init__()
        print("Model creation...")

        self.word2index = w2i
        self.index2word = i2w
        self.max_length = args['maxLengthDeco']

        self.NLLloss = torch.nn.NLLLoss(reduction = 'none')
        self.CEloss =  torch.nn.CrossEntropyLoss(reduction = 'none')

        self.embedding = nn.Embedding(args['vocabularySize'], args['embeddingSize']).to(args['device'])

        self.encoder = Encoder(w2i, i2w, self.embedding).to(args['device'])

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = -1)

        # self.x_2_prob_z = nn.Sequential(
        #     nn.Linear(args['hiddenSize'], 2),
        #     nn.Softmax(dim=-1)
        #   ).to(args['device'])
        self.x_2_prob_z_weight = Parameter(torch.rand(args['chargenum'], args['hiddenSize'], 2)).to(args['device'])
        self.z_to_fea = nn.Sequential(
            nn.Linear(args['hiddenSize'], args['hiddenSize']).to(args['device']),
            nn.Tanh()
          ).to(args['device'])

        self.ChargeClassifier = nn.Sequential(
            nn.Linear(args['hiddenSize'], 1),
            nn.Sigmoid()
          ).to(args['device'])

        '''
        capsule
        '''
        self.cap_Wij = nn.Linear(args['hiddenSize'], args['capsuleSize'],bias=False).to(args['device'])

        # self.z2_mean = Parameter(torch.rand(args['chargenum'], args['capsuleSize'])).to(args['device'])
        # self.z2_logvar = Parameter(torch.rand(args['chargenum'], args['capsuleSize'])).to(args['device'])


        self.q_linear = nn.Linear(args['hiddenSize'], args['hiddenSize']).to(args['device'])
        self.v_linear = nn.Linear(args['hiddenSize'], args['hiddenSize']).to(args['device'])
        self.k_linear = nn.Linear(args['hiddenSize'], args['hiddenSize']).to(args['device'])
        self.z2_hid2mean = nn.Linear(args['hiddenSize'], args['hiddenSize']).to(args['device'])
        self.z2_hid2logvar = nn.Linear(args['hiddenSize'], args['hiddenSize']).to(args['device'])
        
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

    def mask_softmax(self, logits, mask):
        '''
        :param logits: batch seq classnum
        :param mask:  batch seq
        :return: batch seq classnum
        '''
        if len(logits.size()) == 3:
            explogits = torch.exp(logits) * mask.unsqueeze(2)
        elif len(logits.size()) == 2:
            explogits = torch.exp(logits) * mask

        explogits_Z = torch.sum(explogits, dim = 1, keepdim=True)
        softmax = explogits / explogits_Z

        return softmax

    def squash(self, s):
        '''
        :param s:  batch classnum capsuledim
        :return:
        '''
        s_norm = torch.norm(s, dim = 2, keepdim = True)  # batch classnum 1
        v = (s_norm / (1+ s_norm**2)) * s
        return v

    def self_attention(self, q, k, v, d_k, mask=None, dropout=None):
        # k = self.k_linear(k)   # batch seq hid
        # q = self.q_linear(q)
        # v = self.v_linear(v)

        # scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        scores = torch.einsum('bsh,bth->bst', q, k) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill_(mask.unsqueeze(2) == 0, -1e9)  # in place
            scores = scores.masked_fill_(mask.unsqueeze(1) == 0, -1e9)  # in place
        scores = self.softmax(scores)

        if dropout is not None:
            scores = dropout(scores)

        output = torch.einsum('bst,bth->bsh',scores, v)
        return output

    def attention(self, q, mask=None, dropout=None):
        avg_q = torch.sum(q, dim = 1) / (torch.sum(mask.unsqueeze(2), dim = 1)+1)  # batch hid
        scores = torch.einsum('bh,bsh->bs', avg_q,q)

        scores = scores.masked_fill_(mask == 0, -1e9)  # in place
        scores = self.softmax(scores)

        if dropout is not None:
            scores = dropout(scores)

        output = scores.unsqueeze(2) * q # batch s hid
        return output

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

        z_prob = torch.einsum('chy,bsh->bcsy', self.x_2_prob_z_weight, en_outputs)# batch chargenum seq 2
        z_prob = self.softmax(z_prob)

        z_prob_fla = z_prob.reshape((self.batch_size  * args['chargenum']* self.seqlen, 2))
        sampled_seq = self.gumbel_softmax(z_prob_fla).reshape((self.batch_size, args['chargenum'], self.seqlen,  2))
                                                                # batch chargenum seq 2 //0-1
        sampled_seq = sampled_seq * mask.unsqueeze(1).unsqueeze(3)

        # print(sampled_seq)

        sampled_num = torch.sum(sampled_seq[:,:,:,1], dim = 1) # batch chargenum
        sampled_num = (sampled_num == 0).to(args['device'], dtype=torch.float32)  + sampled_num
        sampled_word = en_outputs.unsqueeze(1) * (sampled_seq[:,:,:,1].unsqueeze(3))  # batch  chargenum seq hid
        # s_w_feature = self.z_to_fea(sampled_word)
        # s_w_feature = torch.sum(s_w_feature, dim = 1)/ sampled_num.unsqueeze(1)# batch hid

        '''
        z1 -> z2
        '''
        sampled_word_bc = sampled_word.reshape(self.batch_size  * args['chargenum'], self.seqlen,args['hiddenSize'] )
        z2_words = self.attention(sampled_word_bc,mask=sampled_seq[:,:,:,1].reshape(self.batch_size * args['chargenum'], self.seqlen))
        z2_words = z2_words.reshape(self.batch_size, args['chargenum'], self.seqlen, args['hiddenSize']).to(args['device'])
        z2_hid = z2_words.sum(dim = 2).to(args['device'])  # batch chargenum hid
        z2_mean = self.z2_hid2mean(z2_hid)
        z2_logvar = self.z2_hid2logvar(z2_hid)

        z2 = self.sample_z(z2_mean, z2_logvar) # batch chargenum hid

        # print('z2',torch.sum(z2))


        I_x_z = torch.mean(-torch.log(z_prob[:,:,:,0]+ eps))
        # print(I_x_z)
        # en_hidden, en_cell = en_state   #2 batch hid

        '''
        Capsule
        '''


        capsule_v_norm = self.ChargeClassifier(z2).squeeze()    # b chargenum
        # print('capsule_v_norm: ', capsule_v_norm)

        m_plus = 0.9
        m_minus = 0.1

        cap_pos = self.relu(m_plus - capsule_v_norm) **2  # b chargenum    max(0, *)
        cap_neg = self.relu(capsule_v_norm - m_minus) **2

        answer = F.one_hot(self.classifyLabels, num_classes=args['chargenum']) # batch chargenum
        lambda_c = 0.5

        # print('cap_pos: ', cap_pos.size(), cap_neg.size(), answer.size())
        capsule_loss = answer.float() * cap_pos + lambda_c * (1-answer.float()) * cap_neg
        capsule_loss  = torch.mean(torch.sum(capsule_loss,dim = 1))
        # print('caploss: ',capsule_loss)


        # xz_mock,_ = torch.max(capsule_b ,dim =2 ) # b s
        # xz_mock_p = self.mask_softmax(xz_mock, sampled_seq[:,:,1])
        #
        #
        # # z_regu = - torch.sum(xz_mock_p * torch.log(z_prob[:,:,1]+eps) * sampled_seq[:,:,1], dim = 1)
        # # z_regu = torch.mean(z_regu)
        # xz_mock_p = xz_mock_p.unsqueeze(2) # b s 1
        # diff_xz_mock = torch.triu(xz_mock_p - xz_mock_p.transpose(1,2))
        # pred_z_prob = z_prob[:, :, 1].unsqueeze(2) # b s 1
        # diff_pred_z_prob = torch.triu(pred_z_prob - pred_z_prob.transpose(1,2))
        #
        # # print(diff_xz_mock)
        #
        # z_regu = torch.sum(self.relu(0.1 - diff_xz_mock * diff_pred_z_prob), dim = 2)
        # z_regu = torch.sum(z_regu, dim = 1)
        # z_regu = torch.mean(z_regu)


        # output = self.ChargeClassifier(s_w_feature).to(args['device'])  # batch chargenum
        # recon_loss = self.NLLloss(output, self.classifyLabels).to(args['device'])
        # recon_loss_mean = torch.mean(recon_loss).to(args['device'])

        # print(capsule_loss, z_regu)
        loss = capsule_loss + 0.05 * I_x_z
        return loss

    def predict(self, x):
        encoderInputs = x['enc_input'].to(args['device'])
        encoder_lengths = x['enc_len']

        batch_size = encoderInputs.size()[0]
        seqlen = encoderInputs.size()[1]
        mask = torch.sign(encoderInputs).float()

        en_outputs, en_state = self.encoder(encoderInputs, encoder_lengths)

        z_prob = torch.einsum('chy,bsh->bcsy', self.x_2_prob_z_weight, en_outputs)  # batch chargenum seq 2
        z_prob = self.softmax(z_prob)
        z_prob_fla = z_prob.reshape((batch_size * args['chargenum'] * seqlen, 2))
        sampled_seq = self.gumbel_softmax(z_prob_fla).reshape((batch_size, args['chargenum'], seqlen, 2))
        # batch chargenum seq 2 //0-1
        sampled_seq = sampled_seq * mask.unsqueeze(1).unsqueeze(3)

        # print(sampled_seq)

        sampled_num = torch.sum(sampled_seq[:, :, :, 1], dim=2)  # batch chargenum
        sampled_num = (sampled_num == 0).to(args['device'], dtype=torch.float32) + sampled_num
        sampled_word = en_outputs.unsqueeze(1) * (sampled_seq[:, :, :, 1].unsqueeze(3))

        '''
        z1 -> z2
        '''
        sampled_word_bc = sampled_word.reshape(batch_size * args['chargenum'], seqlen, args['hiddenSize'])
        z2_words = self.attention(sampled_word_bc,mask=sampled_seq[:, :, :, 1].reshape(batch_size * args['chargenum'],seqlen))


        z2_words = z2_words.reshape(batch_size, args['chargenum'], seqlen, args['hiddenSize']).to(args['device'])
        z2_hid = z2_words.sum(dim=2).to(args['device'])  # batch chargenum hid
        z2_mean = self.z2_hid2mean(z2_hid)
        z2_logvar = self.z2_hid2logvar(z2_hid)

        z2 = self.sample_z(z2_mean, z2_logvar)

        '''
        Capsule
        '''
        capsule_v_norm = self.ChargeClassifier(z2)[:,:,0]  # b chargenum

        wordnum = torch.sum(mask, dim = 1, keepdim=True)

        return capsule_v_norm,  (torch.argmax(capsule_v_norm, dim = -1), sampled_seq[:, :, :, 1], sampled_num/wordnum)
