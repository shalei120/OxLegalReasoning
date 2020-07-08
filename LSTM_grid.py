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

class LSTM_grid_Model(nn.Module):
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
        super(LSTM_grid_Model, self).__init__()
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
        self.charge_embs = Parameter(torch.rand(args['chargenum'],args['hiddenSize'])).to(args['device'])

        self.classify2 = nn.Sequential(
            nn.Linear(args['hiddenSize'], 2),
            nn.LogSoftmax(dim=-1)
          ).to(args['device'])
        # self.z_to_fea = nn.Linear(args['hiddenSize'], args['hiddenSize']).to(args['device'])
        # self.ChargeClassifier = nn.Sequential(
        #     nn.Linear(args['hiddenSize'], args['chargenum']),
        #     nn.LogSoftmax(dim=-1)
        #   ).to(args['device'])

        self.indexsequence = torch.LongTensor(list(range(args['maxLength']))).to(args['device'])
        
    def sample_z(self, mu, log_var,batch_size):
        eps = Variable(torch.randn(batch_size, args['style_len']*2* args['numLayers'])).to(args['device'])
        return mu + torch.einsum('ba,ba->ba', torch.exp(log_var/2),eps)

    def forward(self, x):
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

        en_output, en_state = self.encoder(self.encoderInputs, self.encoder_lengths)

        mask = torch.sign(self.encoderInputs).float()
        en_hidden, en_cell = en_state   #2 batch hid

        chargeatt = torch.einsum('bsh,ch->bcs',en_output, self.charge_embs)
        chargeatt = self.softmax(chargeatt)

        indexseq = self.indexsequence[:self.seqlen].float()
        EX = torch.sum(chargeatt * indexseq.unsqueeze(0).unsqueeze(1), dim = 2, keepdim=True) # batch charge 1
        VARX = torch.sum(chargeatt * ((indexseq.unsqueeze(0).unsqueeze(1) - EX)**2), dim = 2) # batch charge
        meanVAR = VARX.mean()
        # print(EX, VARX)
        # exit(0)

        feature_for_each_charge = torch.einsum('bsh,bcs->bch',en_output, chargeatt)

        yesno = self.classify2(feature_for_each_charge) # b c 2

        y = F.one_hot(self.classifyLabels, num_classes=args['chargenum']+2)
        y = y[:,:,:args['chargenum']]
        y = torch.sum(y,dim = 1)  # b chargenum

        recon_loss = yesno[:,:,1] * y.float() + yesno[:,:,0] *(1-y.float())
        recon_loss_mean = torch.mean(torch.sum(-recon_loss, dim = 1)).to(args['device'])

        maxpool,_ = torch.max(en_output, dim = 1) # batch h

        pred = self.sigmoid(maxpool @ self.charge_embs.transpose(0,1)).to(args['device']) #batch c
        loss = y.float() * torch.log(pred) + (1-y.float())*torch.log(1-pred)
        loss = -torch.mean(torch.sum(loss, dim = 1))
        return recon_loss_mean + loss + 0.0001 * meanVAR
        # return recon_loss_mean

    def predict(self, x):
        encoderInputs = x['enc_input']
        encoder_lengths = x['enc_len']

        batch_size = encoderInputs.size()[0]
        enc_len = encoderInputs.size()[1]

        en_output, en_state = self.encoder(encoderInputs, encoder_lengths)

        chargeatt = torch.einsum('bsh,ch->bcs', en_output, self.charge_embs)
        chargeatt = self.softmax(chargeatt)

        feature_for_each_charge = torch.einsum('bsh,bcs->bch', en_output, chargeatt)

        yesno = self.classify2(feature_for_each_charge)  # b c 2

        # choose_res = torch.argmax(yesno, dim = -1)

        choose_res = yesno[:,:,1] > 0.5

        max_choose, _ = torch.max(yesno[:,:,1], dim = 1)
        choose_res = choose_res | (yesno[:,:,1] == max_choose.unsqueeze(1))
        # finalanswer = []
        # for b in range(batch_size):
        #     decode_id_list = []
        #     for ind, choose in enumerate(pred[b,:]):
        #         if choose == 1:
        #             decode_id_list.append(ind)
        #     if len(decode_id_list) == 0:
        #         decode_id_list.append(torch.argmax(yesno[b,:,1]))
        #     finalanswer.append(decode_id_list)


        return choose_res
