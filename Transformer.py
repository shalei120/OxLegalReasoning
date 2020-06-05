import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter

import numpy as np

import datetime, math
from Hyperparameters import args
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x).to(args['device'])

class TransformerModel(nn.Module):
    """
    Implementation of a  model.
    Architecture:
        Encoder/decoder

    """

    def __init__(self, w2i, i2w, ntoken = args['vocabularySize'], ninp = args['embeddingSize'], nhead= 2, nhid = args['hiddenSize'], nlayers = 3, dropout=0.5):
        """
        :param w2i:
        :param i2w:
        :param ntoken:
        :param ninp:
        :param nhead:
        :param nhid:
        :param nlayers:
        :param dropout:
        """
        super(TransformerModel, self).__init__()
        print("Model creation...")

        self.model_type = 'Transformer'

        self.word2index = w2i
        self.index2word = i2w
        self.max_length = args['maxLengthDeco']

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout).to(args['device'])
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout).to(args['device'])
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, args['chargenum'])

        self.init_weights()


        self.NLLloss = torch.nn.NLLLoss(reduction = 'none')
        self.CEloss =  torch.nn.CrossEntropyLoss(reduction = 'none')

        self.embedding = nn.Embedding(args['vocabularySize'], args['embeddingSize']).to(args['device'])


        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = -1)
        self.ChargeClassifier = nn.Sequential(
            nn.Linear(args['hiddenSize'], args['chargenum']),
            nn.LogSoftmax(dim=-1)
        ).to(args['device'])


    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    # def forward(self, src):
    #     if self.src_mask is None or self.src_mask.size(0) != len(src):
    #         device = src.device
    #         mask = self._generate_square_subsequent_mask(len(src)).to(device)
    #         self.src_mask = mask
    #
    #     src = self.encoder(src) * math.sqrt(self.ninp)
    #     src = self.pos_encoder(src)
    #     output = self.transformer_encoder(src)
    #     output = self.decoder(output)
    #     return output

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

        src = self.embedding(self.encoderInputs).to(args['device'])  # batch seq emb
        src = src * math.sqrt(self.ninp)
        src = self.pos_encoder(src).to(args['device'])
        # print(src.shape)
        src_nopad = torch.sign(self.encoderInputs).float()
        output = self.transformer_encoder(src.transpose(0,1)).to(args['device'])  # batch seq hid
        output = output.transpose(0,1)
        # print("output:", output.shape)
        output_avg = torch.mean(output, dim = 1) # batch hid

        chargeprobs = self.ChargeClassifier(output_avg)

        recon_loss = self.NLLloss(chargeprobs, self.classifyLabels).to(args['device'])

        recon_loss_mean = torch.mean(recon_loss).to(args['device'])

        return recon_loss_mean

    def predict(self, x):
        encoderInputs = x['enc_input'].to(args['device'])
        encoder_lengths = x['enc_len']

        src = self.embedding(encoderInputs)   # batch seq emb
        src = src * math.sqrt(self.ninp)
        src = self.pos_encoder(src).to(args['device'])
        output = self.transformer_encoder(src.transpose(0,1)).to(args['device'])    # batch seq hid
        output = output.transpose(0,1)
        output_avg = torch.mean(output, dim = 1) # batch hid

        chargeprobs = self.ChargeClassifier(output_avg)

        return chargeprobs, torch.argmax(chargeprobs, dim = -1)
