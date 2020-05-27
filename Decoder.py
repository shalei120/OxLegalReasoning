import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter

import numpy as np

import datetime
from Hyperparameters import args
from queue import PriorityQueue
import copy

class Decoder(nn.Module):
    def __init__(self, args,w2i, i2w, embedding, device, hidden_size):
        """
        Args:
            args: parameters of the model
            textData: the dataset object
        """
        super(Decoder, self).__init__()
        print("Decoder creation...")

        self.word2index = w2i
        self.index2word = i2w
        self.max_length = args['maxLengthDeco']
        self.device = device

        self.dtype = 'float32'

        self.embedding = embedding

        if args['decunit'] == 'lstm':
            self.dec_unit = nn.LSTM(input_size=args['embeddingSize'],# +  args['numLayers'] *(args['content_len'] + args['style_len']),
                                    hidden_size=args['hiddenSize'],
                                    num_layers=args['dec_numlayer']).to(self.device)
        elif args['decunit'] == 'gru':
            self.dec_unit = nn.GRU(input_size=args['embeddingSize'] +  args['numLayers'] *(args['content_len'] + args['style_len']),
                                   hidden_size=args['hiddenSize'],
                                   num_layers=args['dec_numlayer']).to(self.device)

        self.out_unit = nn.Linear(args['hiddenSize'], args['vocabularySize']).to(self.device)
        self.logsoftmax = nn.LogSoftmax(dim = -1)

        self.element_len = args['hiddenSize']

        self.tanh = nn.Tanh()

    def forward(self, en_state, decoderInputs, decoder_lengths, decoderTargets):
        self.decoderInputs = decoderInputs.to(self.device)
        self.decoder_lengths = decoder_lengths
        self.decoderTargets = decoderTargets.to(self.device)

        self.batch_size = self.decoderInputs.size()[0]
        self.dec_len = self.decoderInputs.size()[1]

        # en_state = (torch.rand(args['dec_numlayer'], self.batch_size, args['hiddenSize'], device = self.device),
        #             torch.rand(args['dec_numlayer'], self.batch_size, args['hiddenSize'], device = self.device))

        dec_input_embed = self.embedding(self.decoderInputs).to(self.device)
        #
        # h,c = torch.split(sentence_emb, args['hiddenSize'], dim = -1)
        #
        # h = self.tanh(self.adapt_unit(h))
        # c = self.tanh(self.adapt_unit(c))
        #
        # sentence_emb = torch.transpose(sentence_emb,0,1).contiguous()
        # sentence_emb = sentence_emb.view(self.batch_size, -1 )

        # sentence_emb_cat = torch.unsqueeze(sentence_emb, 1).repeat(1,self.decoderInputs.size()[1],1)
        # # print('se:',sentence_emb_cat.size())
        #
        # h,c = torch.split(sentence_emb, args['dec_numlayer']*args['hiddenSize'], dim = -1)

        # h,c = torch.split(sentence_emb, args['hiddenSize'], dim = -1)

        # print(h.size(), c.size(), (args['dec_numlayer'], self.batch_size, args['hiddenSize']))

        # h = torch.transpose(h, 0,1)
        # c = torch.transpose(c, 0,1)
        # en_state = (h.contiguous().view(args['dec_numlayer'], self.batch_size, args['hiddenSize']),
        #             c.contiguous().view(args['dec_numlayer'], self.batch_size, args['hiddenSize']))

        # de_outputs, de_state = self.decoder_t(en_state, torch.cat([dec_input_embed, sentence_emb_cat], dim = -1), self.batch_size)
        de_outputs, de_state = self.decoder_t(en_state, dec_input_embed, self.batch_size)

        # de_outputs = self.softmax(de_outputs)
        return de_outputs

    def generate(self, en_state, input_state):

        self.batch_size = en_state[0].size()[1]

        # en_state = (torch.rand(args['dec_numlayer'], self.batch_size, args['hiddenSize'], device = self.device),
        #             torch.rand(args['dec_numlayer'], self.batch_size, args['hiddenSize'], device = self.device))

        # sentence_emb = self.tanh(self.adapt_unit(sentence_emb))
        # sentence_emb = torch.transpose(sentence_emb,0,1).contiguous()
        # sentence_emb = sentence_emb.view(self.batch_size, -1 )
        # h,c = torch.split(sentence_emb, args['dec_numlayer']*args['hiddenSize'], dim = -1)

        # h,c = torch.split(sentence_emb, args['hiddenSize'], dim = -1)
        #
        # h = self.tanh(self.adapt_unit(h))
        # c = self.tanh(self.adapt_unit(c))
        #
        # en_state = (h.contiguous().view(args['dec_numlayer'], self.batch_size, args['hiddenSize']),
        #             c.contiguous().view(args['dec_numlayer'], self.batch_size, args['hiddenSize']))

        de_words = self.decoder_g(en_state)
        for k in range(len(de_words)):
            if '<EOS>' in de_words[k]:
                ind = de_words[k].index('<EOS>')
                de_words[k] = de_words[k][:ind]
        return de_words

    def generate_beam(self, en_state):
        de_words = self.decoder_g_beam(en_state)
        return de_words

    def decoder_t(self, initial_state, inputs, batch_size ):
        inputs = torch.transpose(inputs, 0,1).contiguous()
        state = initial_state

        output, out_state = self.dec_unit(inputs, state)
        # output = output.cpu()

        output = self.out_unit(output.view(batch_size * self.dec_len, args['hiddenSize']))
        output = output.view(self.dec_len, batch_size, args['vocabularySize'])
        output = torch.transpose(output, 0,1)
        return output, out_state

    def decoder_g(self, initial_state):
        state = initial_state
        # sentence_emb = sentence_emb.view(self.batch_size,1, -1 )

        decoded_words = []
        decoder_input_id = torch.tensor([[self.word2index['START_TOKEN'] for _ in range(self.batch_size)]], device=self.device)  # SOS 1*batch
        decoder_input = self.embedding(decoder_input_id).contiguous().to(self.device)
        # print('decoder input: ', decoder_input.shape)
        decoder_id_res = []
        for di in range(self.max_length):

            # decoder_output, state = self.dec_unit(torch.cat([decoder_input, sentence_emb], dim = -1), state)
            decoder_output, state = self.dec_unit(decoder_input, state)

            decoder_output = self.out_unit(decoder_output)

            topv, topi = decoder_output.data.topk(1, dim = -1)

            decoder_input_id = topi[:,:,0].detach()
            decoder_id_res.append(decoder_input_id)
            decoder_input = self.embedding(decoder_input_id).to(self.device)

        decoder_id_res = torch.cat(decoder_id_res, dim = 0)  #seqlen * batch

        for b in range(self.batch_size):
            decode_id_list = list(decoder_id_res[:,b])
            if self.word2index['END_TOKEN'] in decode_id_list:
                decode_id_list = decode_id_list[:decode_id_list.index(self.word2index['END_TOKEN'])] if decode_id_list[0] != self.word2index['END_TOKEN'] else [self.word2index['END_TOKEN']]
            decoded_words.append([self.index2word[id] for id in decode_id_list])
        return decoded_words

    def decoder_g_beam(self, initial_state, beam_width = 10):
        parent = self
        class Subseq:
            def __init__(self ):
                self.logp = 0.0
                self.sequence = [parent.word2index['START_TOKEN']]

            def append(self, wordindex, logp):
                self.sequence.append(wordindex)
                self.logp += logp

            def append_createnew(self, wordindex, logp):
                newss = copy.deepcopy(self)
                newss.sequence.append(wordindex)
                newss.logp += logp
                return newss

            def eval(self):
                return self.logp / float(len(self.sequence) - 1 + 1e-6)

            def __lt__(self, other): # add negative
                return self.eval() > other.eval()

        state = initial_state
        pq = PriorityQueue()
        pq.put(Subseq())

        for di in range(self.max_length):
            pqitems = []
            for _ in range(beam_width):
                pqitems.append(pq.get())
                if pq.empty():
                    break
            pq.queue.clear()

            end = True
            for subseq in pqitems:
                if subseq.sequence[-1] == self.word2index['END_TOKEN']:
                    pq.put(subseq)
                else:
                    end = False
                    lastindex = subseq.sequence[-1]
                    decoder_input_id = torch.tensor([[lastindex]], device=self.device)  # SOS
                    decoder_input = self.embedding(decoder_input_id).contiguous().to(self.device)
                    decoder_output, state = self.dec_unit(decoder_input, state)

                    decoder_output = self.out_unit(decoder_output)
                    decoder_output = self.logsoftmax(decoder_output)

                    logps, indexes = decoder_output.data.topk(beam_width)

                    for logp, index in zip(logps[0][0], indexes[0][0]):
                        newss = subseq.append_createnew(index.item(), logp)
                        pq.put(newss )
            if end:
                break

        finalseq = pq.get()

        decoded_words = []
        for i in finalseq.sequence[1:]:
            decoded_words.append(self.index2word[i])

        return decoded_words











