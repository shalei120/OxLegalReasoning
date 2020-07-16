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

from textdata import TextData
from kenLM import LMEvaluator as LMEr

class LanguageModel(nn.Module):
    def __init__(self,w2i, i2w):
        """
        Args:
            args: parameters of the model
            textData: the dataset object
        """
        super(LanguageModel, self).__init__()
        print("LanguageModel creation...")

        self.word2index = w2i
        self.index2word = i2w
        self.max_length = args['maxLengthDeco']
        self.device = args['device']
        self.batch_size = args['batchSize']

        self.dtype = 'float32'

        self.embedding = nn.Embedding(args['vocabularySize'], args['embeddingSize']).to(self.device)

        if args['decunit'] == 'lstm':
            self.dec_unit = nn.LSTM(input_size=args['embeddingSize'],
                                    hidden_size=args['hiddenSize'],
                                    num_layers=args['dec_numlayer']).to(self.device)
        elif args['decunit'] == 'gru':
            self.dec_unit = nn.GRU(input_size=args['embeddingSize'],
                                   hidden_size=args['hiddenSize'],
                                   num_layers=args['dec_numlayer']).to(self.device)

        self.out_unit = nn.Linear(args['hiddenSize'], args['vocabularySize']).to(self.device)
        self.logsoftmax = nn.LogSoftmax(dim = -1)

        self.element_len = args['hiddenSize']

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = -1)
        self.CEloss = torch.nn.CrossEntropyLoss(reduction='none')

        self.init_state = (torch.rand(args['dec_numlayer'], 1, args['hiddenSize'], device=self.device),
                           torch.rand(args['dec_numlayer'], 1, args['hiddenSize'], device=self.device))

    def buildModel(self, x):
        self.decoderInputs = x['dec_input'].to(self.device)
        self.decoder_lengths = x['dec_len']
        self.decoderTargets = x['dec_target'].to(self.device)

        batch_size = self.decoderInputs.size()[0]
        self.dec_len = self.decoderInputs.size()[1]

        dec_input_embed = self.embedding(self.decoderInputs).to(self.device)

        init_state = (self.init_state[0].repeat([1, batch_size, 1]), self.init_state[1].repeat([1, batch_size, 1]))

        de_outputs, de_state = self.decoder_t(init_state, dec_input_embed, batch_size, self.dec_len)

        recon_loss = self.CEloss(torch.transpose(de_outputs, 1, 2), self.decoderTargets)
        mask = torch.sign(self.decoderTargets.float())
        recon_loss = torch.squeeze(recon_loss) * mask

        recon_loss_mean = torch.mean(recon_loss, dim=-1)
        return de_outputs, recon_loss_mean

    def decoder_t(self, initial_state, inputs, batch_size, dec_len):
        inputs = torch.transpose(inputs, 0,1).contiguous()
        state = initial_state

        output, out_state = self.dec_unit(inputs, state)
        # output = output.cpu()

        output = self.out_unit(output.view(batch_size * dec_len, args['hiddenSize']))
        output = output.view(dec_len, batch_size, args['vocabularySize'])
        output = torch.transpose(output, 0,1)
        return output, out_state

    def forward(self, x):
        de_outputs, recon_loss_mean = self.buildModel(x)
        return de_outputs, recon_loss_mean

def train(textData, model, model_path, print_every=10000, plot_every=10, learning_rate=0.001):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    print(type(self.textData.word2index))

    optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-3, amsgrad=True)

    iter = 1
    batches = textData.getBatches()
    n_iters = len(batches)
    print('niters ', n_iters)

    args['trainseq2seq'] = False

    min_ppl = -1
    # accuracy = self.test('test', max_accu)
    for epoch in range(args['numEpochs']):
        losses = []

        for batch in batches:
            optimizer.zero_grad()
            x = {}
            x['dec_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs)).to(args['device'])
            x['dec_len'] = batch.encoder_lens
            x['dec_target'] = x['dec_input'][:,1:]
            x['dec_input'] = x['dec_input'][:,:-1]

            _, loss = self.model(x)  # batch seq_len outsize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args['clip'])

            optimizer.step()

            print_loss_total += loss.data
            plot_loss_total += loss.data

            losses.append(loss.data)

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / (n_iters * args['numEpochs'])),
                                                 iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

            iter += 1

        ppl = self.test('test')
        if ppl < min_ppl or min_ppl == -1:
            print('ppl = ', ppl, '<= min_ppl(', min_ppl, '), saving model...')
            torch.save(model, model_path)
            min_ppl = ppl

        print('Epoch ', epoch, 'loss = ', sum(losses) / len(losses), 'Valid ppl = ', ppl,
              'min ppl=', min_ppl)

    # self.test()
    # showPlot(plot_losses)

def test(textData, model, datasetname, eps=1e-20):
    ave_loss = 0
    num = 0
    with torch.no_grad():
        for batch in textData.getBatches(datasetname):
            x = {}
            x['dec_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs)).to(args['device'])
            x['dec_len'] = batch.encoder_lens
            x['dec_target'] = x['dec_input'][:,1:]
            x['dec_input'] = x['dec_input'][:,:-1]
            _, recon_loss_mean = model(x)  # batch seq_len outsize
            ave_loss = (ave_loss * num + sum(recon_loss_mean)) / (num + len(recon_loss_mean))
            num += len(recon_loss_mean)

    return torch.exp(ave_loss)



if __name__ == '__main__':
    textData = TextData('cail')
    args['vocabularySize'] = self.textData.getVocabularySize()
    args['chargenum'] = self.textData.getChargeNum()
    model = LanguageModel(textData.word2index, textData.index2word)
    train(textData, model, model_path = args['rootDir']+'/LM.pkl')
