# Copyright 2020 . All Rights Reserved.
# Author : Lei Sha

import argparse
import os

from textdata import TextData
import time, sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import  tqdm
import time
import math,random
import nltk
import pickle
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
# import matplotlib.pyplot as plt
import numpy as np
import copy
from Hyperparameters import args
from LSTM import LSTM_Model
from LSTM_IB import LSTM_IB_Model
from Transformer import TransformerModel

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g')
parser.add_argument('--modelarch', '-m')
cmdargs = parser.parse_args()

usegpu = True

if cmdargs.gpu is None:
    usegpu = False
else:
    usegpu = True
    args['device'] = 'cuda:' + str(cmdargs.gpu)

if cmdargs.modelarch is None:
    args['model_arch'] = 'lstm'
else:
    args['model_arch'] = cmdargs.modelarch



def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

class Runner:
    def __init__(self):
        self.model_path = args['rootDir'] + '/chargemodel_' + args['model_arch']+ '.mdl'

    def main(self):
        self.textData = TextData('cail')
        self.start_token = self.textData.word2index['START_TOKEN']
        self.end_token = self.textData.word2index['END_TOKEN']
        args['vocabularySize'] = self.textData.getVocabularySize()
        args['chargenum'] = self.textData.getChargeNum()
        print(self.textData.getVocabularySize())

        if args['model_arch'] == 'lstm':
            print('Using LSTM model.')
            self.model = LSTM_Model(self.textData.word2index, self.textData.index2word)
        elif args['model_arch'] == 'transformer':
            print('Using Transformer model.')
            self.model = TransformerModel(self.textData.word2index, self.textData.index2word)
        elif args['model_arch'] == 'lstmib':
            print('Using LSTM information bottleneck model.')
            self.model = LSTM_IB_Model(self.textData.word2index, self.textData.index2word)

        self.train()

    def train(self, print_every=10000, plot_every=10, learning_rate=0.001):
        start = time.time()
        plot_losses = []

        print(type(self.textData.word2index))

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-3, amsgrad=True)

        iter = 1
        batches = self.textData.getBatches()
        n_iters = len(batches)
        print('niters ', n_iters)

        args['trainseq2seq'] = False

        min_accu = -1

        for epoch in range(args['numEpochs']):
            losses = []

            for batch in batches:
                optimizer.zero_grad()
                x = {}
                x['enc_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs)).to(args['device'])
                x['enc_len'] = batch.encoder_lens
                x['labels'] = autograd.Variable(torch.LongTensor(batch.label)).to(args['device'])

                loss = self.model(x)  # batch seq_len outsize

                loss.backward(retain_graph=True)

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

            accuracy = self.test('test')
            if accuracy < min_accu or min_accu == -1:
                print('accuracy = ', accuracy, '>= min_accuracy(', min_accu, '), saving model...')
                torch.save(self.model, self.model_path)
                min_accu = accuracy

            print('Epoch ', epoch, 'loss = ', sum(losses) / len(losses), 'Valid accuracy = ', accuracy)

        # self.test()
        # showPlot(plot_losses)

    def test(self, datasetname):
        if not hasattr(self, 'testbatches'):
            self.testbatches = {}
        if datasetname not in self.testbatches:
            self.testbatches[datasetname] = self.textData.getBatches(datasetname)
        right = 0
        total = 0

        dset = []

        with torch.no_grad():
            for batch in self.testbatches[datasetname]:
                x = {}
                x['enc_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs))
                x['enc_len'] = batch.encoder_lens

                output_probs, output_labels = self.model.predict(x)

                batch_correct = output_labels.cpu().numpy() == torch.LongTensor(batch.label).cpu().numpy()
                right += sum(batch_correct)
                total += x['enc_input'].size()[0]

                for ind, c in enumerate(batch_correct):
                    if not c:
                        dset.append((batch.encoderSeqs[ind], batch.label[ind], output_labels[ind]))

        accuracy = right / total

        with open(args['rootDir'] + '/error_case_'+args['model_arch']+'.txt', 'w') as wh:
            for d in dset:
                wh.write(''.join([self.textData.index2word[wid] for wid in d[0]]))
                wh.write('\t')
                wh.write(self.textData.lawinfo['i2c'][int(d[1])])
                wh.write('\t')
                wh.write(self.textData.lawinfo['i2c'][int(d[2])])
                wh.write('\n')
        wh.close()

        return accuracy

    def indexesFromSentence(self,  sentence):
        return [self.textData.word2index[word] if word in self.textData.word2index else self.textData.word2index['UNK'] for word in sentence]

    def tensorFromSentence(self, sentence):
        indexes = self.indexesFromSentence(sentence)
        # indexes.append(self.textData.word2index['END_TOKEN'])
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

    def evaluate(self,  sentence, correctlabel, max_length=20):
        with torch.no_grad():
            input_tensor = self.tensorFromSentence( sentence)
            input_length = input_tensor.size()[0]
            # encoder_hidden = encoder.initHidden()

            # encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

            x={}
            # print(input_tensor)
            x['enc_input'] = torch.transpose(input_tensor, 0,1)
            x['enc_len'] = [input_length]
            x['labels'] = [correctlabel]
            # print(x['enc_input'], x['enc_len'])
            # print(x['enc_input'].shape)
            decoded_words, label,_ = self.model.predict(x, True)

            return decoded_words, label

    def evaluateRandomly(self, n=10):
        for i in range(n):
            sample = random.choice(self.textData.datasets['train'])
            print('>', sample)
            output_words, label = self.evaluate(sample[2], sample[1])
            output_sentence = ' '.join(output_words[0])  # batch=1
            print('<', output_sentence, label)
            print('')


if __name__ == '__main__':
    
    r = Runner()
    r.main()