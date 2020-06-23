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

class LSTM_capsule_Model(nn.Module):
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
        super(LSTM_capsule_Model, self).__init__()
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

        self.x_2_prob_z = nn.Sequential(
            nn.Linear(args['hiddenSize'], 2),
            nn.Softmax(dim=-1)
          ).to(args['device'])
        self.z_to_fea = nn.Linear(args['hiddenSize'], args['hiddenSize']).to(args['device'])

        self.ChargeClassifier = nn.Sequential(
            nn.Linear(args['hiddenSize'], args['chargenum']),
            nn.LogSoftmax(dim=-1)
          ).to(args['device'])

        '''
        capsule
        '''
        self.cap_Wij = nn.Linear(args['hiddenSize'], 50,bias=False).to(args['device'])
        


    def mask_softmax(self, logits, mask):
        '''
        :param logits: batch seq classnum
        :param mask:  batch seq
        :return: batch seq classnum
        '''

        explogits = torch.exp(logits) * mask.unsqueeze(2)
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
        '''
        Capsule
        '''
        capsule_uji = self.cap_Wij(en_outputs)  # b s cap
        capsule_b = Parameter(torch.zeros(self.batch_size, self.seqlen, args['chargenum'])).to(args['device'])

        for _ in range(3):
            capsule_c = self.mask_softmax(capsule_b, mask)  # b s chargenum
            capsule_s = torch.einsum('bsc,bsn->bnc', capsule_uji, capsule_c)
            capsule_v = self.squash(capsule_s) # b chargenum cap
            capsule_delta = torch.einsum('bcp,bsp->bsc', capsule_v, capsule_uji) # b s chargenum
            capsule_b = capsule_delta


        capsule_v_norm = torch.norm(capsule_v, dim = 2)# b chargenum

        m_plus = 0.9
        m_minus = 0.1

        cap_pos = self.relu(m_plus - capsule_v_norm) **2  # b chargenum    max(0, *)
        cap_neg = self.relu(capsule_v_norm - m_minus) **2

        answer = F.one_hot(self.classifyLabels, num_classes=args['chargenum']) # batch chargenum
        lambda_c = 0.5
        capsule_loss = answer.float() * cap_pos + lambda_c * (1-answer.float()) * cap_neg

        loss = torch.sum(capsule_loss, dim = 1)

        # self.final = Parameter(torch.rand(50)).to(args['device'])
        # pred_logit = torch.einsum('bcp,p->bc', capsule_v, self.final)
        # pred = self.softmax(pred_logit)
        # loss = self.CEloss(pred, self.classifyLabels).to(args['device'])
        return torch.mean(loss)

    def predict(self, x):
        encoderInputs = x['enc_input'].to(args['device'])
        encoder_lengths = x['enc_len']

        batch_size = encoderInputs.size()[0]
        seqlen = encoderInputs.size()[1]
        mask = torch.sign(encoderInputs).float()

        en_outputs, en_state = self.encoder(encoderInputs, encoder_lengths)

        '''
        Capsule
        '''
        capsule_uji = self.cap_Wij(en_outputs)  # b s cap
        capsule_b = torch.zeros(batch_size, seqlen, args['chargenum']).to(args['device'])

        for _ in range(3):
            capsule_c = self.mask_softmax(capsule_b, mask)  # b s chargenum
            capsule_s = torch.einsum('bsc,bsn->bnc', capsule_uji, capsule_c)
            capsule_v = self.squash(capsule_s)  # b chargenum cap
            capsule_delta = torch.einsum('bcp,bsp->bsc', capsule_v, capsule_uji)  # b s chargenum
            capsule_b += capsule_delta.detach()

        capsule_v_norm = torch.norm(capsule_v, dim=2)  # b chargenum

        # pred_logit = torch.einsum('bcp,p->bc', capsule_v, self.final)


        return capsule_v_norm,  torch.argmax(capsule_v_norm, dim = -1)
        # return pred_logit,  torch.argmax(pred_logit, dim = -1)
