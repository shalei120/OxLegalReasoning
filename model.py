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

class Model(nn.Module):
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
        super(Model, self).__init__()
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

        self.ChargeClassifier = nn.Sequential(
            nn.Linear(args['hiddenSize'] * args['numLayers'], args['chargenum']),
            nn.LogSoftmax(dim=-1)
          ).to(args['device'])
        
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

        en_state = self.encoder(self.encoderInputs, self.encoder_lengths)

        en_hidden, en_cell = en_state   #2 batch hid

        output = self.ChargeClassifier(en_hidden.transpose(0,1).reshape(self.batch_size,-1)).to(args['device'])  # batch chargenum

        recon_loss = self.NLLloss(output, self.classifyLabels).to(args['device'])

        recon_loss_mean = torch.mean(recon_loss).to(args['device'])

        return recon_loss_mean

    def predict(self, x):
        encoderInputs = x['enc_input']
        encoder_lengths = x['enc_len']

        batch_size = encoderInputs.size()[0]
        enc_len = encoderInputs.size()[1]

        en_state = self.encoder(encoderInputs, encoder_lengths)

        en_hidden, en_cell = en_state  # batch hid

        output = self.ChargeClassifier(en_hidden.transpose(0,1).reshape(batch_size,-1)).to(args['device']) # batch chargenum


        return output, torch.argmax(output, dim = -1)
