# Copyright 2020 . All Rights Reserved.
# Author : Lei Sha

from Hyperparameters import args
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g')
parser.add_argument('--modelarch', '-m')
parser.add_argument('--aspect', '-a')
parser.add_argument('--choose', '-c')
cmdargs = parser.parse_args()
print(cmdargs)
usegpu = True
if cmdargs.gpu is None:
    usegpu = False

    args['device'] = 'cpu'
else:
    usegpu = True
    args['device'] = 'cuda:' + str(cmdargs.gpu)

if cmdargs.modelarch is None:
    args['model_arch'] = 'lstm'
else:
    args['model_arch'] = cmdargs.modelarch

if cmdargs.aspect is None:
    args['aspect'] = 0
else:
    args['aspect'] = int(cmdargs.aspect)
if cmdargs.choose is None:
    args['choose'] = 0
else:
    args['choose'] = int(cmdargs.aspect)

import functools
print = functools.partial(print, flush=True)
import os

from textdataBeer import TextDataBeer
import time, sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import time, datetime
import math, random
import nltk
import pickle
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
# import matplotlib.pyplot as plt
import numpy as np
import copy
from LanguageModel_beer import LanguageModel

import LSTM_IB_GAN_beer


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (%s)' % (asMinutes(s), datetime.datetime.now())


class Runner:
    def __init__(self):
        self.model_path = args['rootDir'] + '/chargemodel_' + args['model_arch'] + '.mdl'

    def main(self):

        if args['model_arch'] in ['lstmibgan']:
            args['classify_type'] = 'single'
            args['batchSize'] = 256

        self.textData = TextDataBeer('beer')
        # self.start_token = self.textData.word2index['START_TOKEN']
        # self.end_token = self.textData.word2index['END_TOKEN']
        args['vocabularySize'] = self.textData.getVocabularySize()
        args['chargenum'] = 5
        args['embeddingSize'] = self.textData.index2vector.shape[1]
        print(self.textData.getVocabularySize())
        args['model_arch'] = 'lstmibgan'
        # args['aspect'] = 0
        args['hiddenSize'] = 200

        print(args)
        if args['model_arch'] == 'lstmibgan':
            print('Using LSTM information bottleneck GAN model for Beer.')
            LM = torch.load(args['rootDir']+'/LMbeer.pkl', map_location=args['device'])
            for param in LM.parameters():
                param.requires_grad = False

            ppl = self.CalPPL(LM)
            print('PPL=',ppl)
            # LM=0
            LSTM_IB_GAN_beer.train(self.textData, LM, self.textData.index2vector)

    def indexesFromSentence(self, sentence):
        return [self.textData.word2index[word] if word in self.textData.word2index else self.textData.word2index['UNK']
                for word in sentence]

    def tensorFromSentence(self, sentence):
        indexes = self.indexesFromSentence(sentence)
        # indexes.append(self.textData.word2index['END_TOKEN'])
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

    def evaluate(self, sentence, correctlabel, max_length=20):
        with torch.no_grad():
            input_tensor = self.tensorFromSentence(sentence)
            input_length = input_tensor.size()[0]
            # encoder_hidden = encoder.initHidden()

            # encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

            x = {}
            # print(input_tensor)
            x['enc_input'] = torch.transpose(input_tensor, 0, 1)
            x['enc_len'] = [input_length]
            x['labels'] = [correctlabel]
            # print(x['enc_input'], x['enc_len'])
            # print(x['enc_input'].shape)
            decoded_words, label, _ = self.model.predict(x, True)

            return decoded_words, label

    def evaluateRandomly(self, n=10):
        for i in range(n):
            sample = random.choice(self.textData.datasets['train'])
            print('>', sample)
            output_words, label = self.evaluate(sample[2], sample[1])
            output_sentence = ' '.join(output_words[0])  # batch=1
            print('<', output_sentence, label)
            print('')

    def CalPPL(self, LM):

        batches = self.textData.getBatches('dev')
        total = 0
        loss_sum = 0
        for index, batch in enumerate(batches):
            x = {}
            x['dec_input'] = autograd.Variable(torch.LongTensor(batch.decoderSeqs)).to(args['device'])
            x['dec_len'] = batch.decoder_lens
            x['dec_target'] = autograd.Variable(torch.LongTensor(batch.targetSeqs)).to(args['device'])
            total += x['dec_input'].size()[0]
            print(x['dec_input'].size())

            embedding = nn.Embedding.from_pretrained(torch.FloatTensor(self.textData.index2vector))
            decoderTargetsEmbeddings = embedding(x['dec_target'])
            _, recon_loss = LM.getloss(x['dec_input'],decoderTargetsEmbeddings, x['dec_target']  )
            loss_sum += recon_loss.sum()

        loss_mean = loss_sum / total
        return torch.exp(loss_mean)


if __name__ == '__main__':

    r = Runner()
    r.main()