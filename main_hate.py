# Copyright 2020 . All Rights Reserved.
# Author : Lei Sha
import functools
print = functools.partial(print, flush=True)
import argparse
import os

from textdataHate import TextDataHate
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
from Hyperparameters import args
from LanguageModel import LanguageModel

import LSTM_IB_GAN_hate

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g')
parser.add_argument('--modelarch', '-m')
parser.add_argument('--choose', '-c')
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


if cmdargs.choose is None:
    args['choose'] = 0
else:
    args['choose'] = int(cmdargs.choose)

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
        self.model_path = args['rootDir'] + '/hatespeechmodel_' + args['model_arch'] + '.mdl'

    def main(self):

        self.textData = TextDataHate('hate')
        self.start_token = self.textData.word2index['[START_TOKEN]']
        self.end_token = self.textData.word2index['[END_TOKEN]']
        args['vocabularySize'] = self.textData.getVocabularySize()

        args['hiddenSize'] = 200

        print(self.textData.getVocabularySize())

        # if args['model_arch'].startswith('lstmibgan'):
        print('Using LSTM information bottleneck GAN model. Task: hatespeech' )
        # LM = torch.load(args['rootDir']+'/LM'+ args['datasetsize'] +'.pkl', map_location=args['device'])
        # for param in LM.parameters():
        #     param.requires_grad = False

        LSTM_IB_GAN_hate.train(self.textData)


    def indexesFromSentence(self, sentence):
        return [self.textData.word2index[word] if word in self.textData.word2index else self.textData.word2index['[UNK]']
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


if __name__ == '__main__':
    r = Runner()
    r.main()