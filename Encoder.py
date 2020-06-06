import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import datetime
from Hyperparameters import args

class Encoder(nn.Module):
    def __init__(self,w2i, i2w, embedding):
        """
        Args:
            args: parameters of the model
            textData: the dataset object
        """
        super(Encoder, self).__init__()
        print("Encoder creation...")

        self.word2index = w2i
        self.index2word = i2w
        self.max_length = args['maxLengthDeco']

        self.dtype = 'float32'

        self.embedding = embedding

        if args['encunit'] == 'lstm':
            self.enc_unit = nn.LSTM(input_size=args['embeddingSize'], hidden_size=args['hiddenSize'],
                                    num_layers=args['enc_numlayer']).to(args['device'])
        elif args['encunit'] == 'gru':
            self.enc_unit = nn.GRU(input_size=args['embeddingSize'], hidden_size=args['hiddenSize'],
                                   num_layers=args['enc_numlayer']).to(args['device'])

        self.element_len = args['hiddenSize']


    def forward(self, encoderInputs, encoder_lengths):
        '''
        :param encoderInputs: [batch, enc_len]
        :param decoderInputs: [batch, dec_len]
        :param decoderTargets: [batch, dec_len]
        :return:
        '''

        # print(x['enc_input'])
        self.encoderInputs = encoderInputs.to(args['device'])
        self.encoder_lengths = encoder_lengths

        self.batch_size = self.encoderInputs.size()[0]
        self.enc_len = self.encoderInputs.size()[1]

        enc_input_embed = self.embedding(self.encoderInputs).to(args['device'])#.cuda()   # batch enc_len embedsize  ; already sorted in a decreasing order
        # dec_target_embed = self.embedding(self.decoderTargets).cuda()   # batch dec_len embedsize

        en_outputs, en_state = self.encode(enc_input_embed, self.batch_size) # seq batch emb

        # if np.isnan(en_outputs.data).any()>0:
        #     sdf=0

        # en_state_cat = torch.cat(en_state, dim = -1).to(self.device)

        en_outputs = torch.transpose(en_outputs, 0, 1)

        return en_outputs, en_state

    def encode(self, inputs, batch_size):
        inputs = torch.transpose(inputs, 0, 1)
        hidden = (
        autograd.Variable(torch.randn(args['enc_numlayer'], batch_size, args['hiddenSize'])).to(args['device']),
        autograd.Variable(torch.randn(args['enc_numlayer'], batch_size, args['hiddenSize'])).to(args['device']))
        # print('sdfw',inputs.shape, self.batch_size)
        # packed_input = nn.utils.rnn.pack_padded_sequence(inputs, input_len)
        packed_input = inputs
        packed_out, hidden = self.enc_unit(packed_input, hidden)
        # if np.isnan(packed_out.data).any()>0:
        #     sdf=0
        return packed_out, hidden
