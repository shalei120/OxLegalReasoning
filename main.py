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
from LSTM_att import LSTM_att_Model
from LSTM_IB import LSTM_IB_Model
import LSTM_IB_GAN
from LSTM_IB_complete import LSTM_IB_CP_Model
from Transformer import TransformerModel
from LSTM_capIB import LSTM_capsule_IB_Model
from LSTM_cap import LSTM_capsule_Model
from LSTM_iterIB import LSTM_iterIB_Model
from LSTM_grid import LSTM_grid_Model
from LSTM_GMIB import LSTM_GMIB_Model

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
            self.train()
        elif args['model_arch'] == 'lstmatt':
            print('Using LSTM attention model.')
            self.model = LSTM_att_Model(self.textData.word2index, self.textData.index2word)
            self.train()
        elif args['model_arch'] == 'transformer':
            print('Using Transformer model.')
            self.model = TransformerModel(self.textData.word2index, self.textData.index2word)
            self.train()
        elif args['model_arch'] == 'lstmib':
            print('Using LSTM information bottleneck model.')
            self.model = LSTM_IB_Model(self.textData.word2index, self.textData.index2word)
            self.train()
        elif args['model_arch'] == 'lstmibgan':
            print('Using LSTM information bottleneck GAN model.')
            LSTM_IB_GAN.train(self.textData)
        elif args['model_arch'] == 'lstmibcp':
            print('Using LSTM information bottleneck model. -- complete words')
            self.model = LSTM_IB_CP_Model(self.textData.word2index, self.textData.index2word)
            self.train()
        elif args['model_arch'] == 'lstmcapib':
            print('Using LSTM capsule information bottleneck model.')
            self.model = LSTM_capsule_IB_Model(self.textData.word2index, self.textData.index2word)
            self.train()
        elif args['model_arch'] == 'lstmiterib':
            print('Using LSTM iteratively information bottleneck model.')
            self.model = LSTM_iterIB_Model(self.textData.word2index, self.textData.index2word)
            self.train()
        elif args['model_arch'] == 'lstmcap':
            print('Using LSTM capsule model.')
            self.model = LSTM_capsule_Model(self.textData.word2index, self.textData.index2word)
            self.train()
        elif args['model_arch'] == 'lstmgrid':
            print('Using LSTM grid model.')
            self.model = LSTM_grid_Model(self.textData.word2index, self.textData.index2word)
            self.train()
        elif args['model_arch'] == 'lstmgmib':
            print('Using LSTM Gaussian Mixture IB model.')
            self.model = LSTM_GMIB_Model(self.textData.word2index, self.textData.index2word)
            self.train()


    def train(self, print_every=10000, plot_every=10, learning_rate=0.001):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        print(type(self.textData.word2index))

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-3, amsgrad=True)

        iter = 1
        batches = self.textData.getBatches()
        n_iters = len(batches)
        print('niters ', n_iters)

        args['trainseq2seq'] = False

        max_accu = -1
        # accuracy = self.test('test', max_accu)
        for epoch in range(args['numEpochs']):
            losses = []

            for batch in batches:
                optimizer.zero_grad()
                x = {}
                x['enc_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs)).to(args['device'])
                x['enc_len'] = batch.encoder_lens
                x['labels'] = autograd.Variable(torch.LongTensor(batch.label)).to(args['device'])

                if  args['model_arch'] not in ['lstmiterib', 'lstmgrid']:
                    x['labels'] = x['labels'][:,0]

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
            if args['model_arch'] == 'lstmiterib':
                accuracy, EM, p,r,acc = self.test('test', max_accu)
                if accuracy > max_accu or max_accu == -1:
                    print('accuracy = ', accuracy, '>= min_accuracy(', max_accu, '), saving model...')
                    torch.save(self.model, self.model_path)
                    max_accu = accuracy

                print('Epoch ', epoch, 'loss = ', sum(losses) / len(losses), 'Valid accuracy = ', accuracy,EM, p,r,acc,
                      'max accuracy=', max_accu)

            else:
                accuracy = self.test('test', max_accu)
                if accuracy > max_accu or max_accu == -1:
                    print('accuracy = ', accuracy, '>= min_accuracy(', max_accu, '), saving model...')
                    torch.save(self.model, self.model_path)
                    max_accu = accuracy

                print('Epoch ', epoch, 'loss = ', sum(losses) / len(losses), 'Valid accuracy = ', accuracy, 'max accuracy=', max_accu)

        # self.test()
        # showPlot(plot_losses)

    def test(self, datasetname, max_accuracy):
        # if not hasattr(self, 'testbatches'):
        #     self.testbatches = {}
        # if datasetname not in self.testbatches:
        # self.testbatches[datasetname] = self.textData.getBatches(datasetname)
        right = 0
        total = 0

        dset = []

        exact_match = 0
        p = 0.0
        r = 0.0
        acc = 0.0

        with torch.no_grad():
            pppt = False
            for batch in self.textData.getBatches(datasetname):
                x = {}
                x['enc_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs))
                x['enc_len'] = batch.encoder_lens

                if args['model_arch'] in ['lstmiterib', 'lstmgrid']:
                    answerlist = self.model.predict(x)
                    for anses, gold in zip(answerlist, batch.label):
                        anses = [int(ele) for ele in anses]
                        if anses[0] == gold[0]:
                            right+=1
                        goldlist = list(gold[:gold.index(args['chargenum'])])
                        intersect = set(anses)
                        joint = set(anses)
                        intersect = intersect.intersection(set(goldlist))
                        joint.update(set(goldlist))
                        intersect_size=len(intersect)
                        joint_size= len(joint)
                        if intersect_size == joint_size:
                            exact_match += 1

                        # print(intersect,joint, anses, goldlist)

                        acc = (acc * total + intersect_size / joint_size) / (total+1)
                        p = (p * total + intersect_size / len(anses)) / (total+1)
                        r = (r * total + intersect_size / len(goldlist)) / (total+1)

                        # print(acc, p,r)
                        # exit()
                        total+=1

                else:
                    output_probs, output_labels = self.model.predict(x)
                    if  args['model_arch'] == 'lstmib'or args['model_arch'] == 'lstmibcp' :
                        output_labels, sampled_words, wordsamplerate = output_labels
                        if not pppt:
                            pppt = True
                            for w, choice in zip(batch.encoderSeqs[0], sampled_words[0]):
                                if choice[1] == 1:
                                    print(self.textData.index2word[w], end='')
                            print('sample rate: ', wordsamplerate[0])
                    elif args['model_arch'] == 'lstmcapib':
                        output_labels, sampled_words, wordsamplerate = output_labels
                        if not pppt:
                            pppt = True
                            for w, choice in zip(batch.encoderSeqs[0], sampled_words[0,output_labels[0],:]):
                                if choice == 1:
                                    print(self.textData.index2word[w], end='')
                            print('sample rate: ', wordsamplerate[0])
                            
                            
                    batch_correct = output_labels.cpu().numpy() == torch.LongTensor(batch.label).cpu().numpy()
                    right += sum(batch_correct)
                    total += x['enc_input'].size()[0]

                    for ind, c in enumerate(batch_correct):
                        if not c:
                            dset.append((batch.encoderSeqs[ind], batch.label[ind], output_labels[ind]))

        accuracy = right / total

        if accuracy > max_accuracy:
            with open(args['rootDir'] + '/error_case_'+args['model_arch']+'.txt', 'w') as wh:
                for d in dset:
                    wh.write(''.join([self.textData.index2word[wid] for wid in d[0]]))
                    wh.write('\t')
                    wh.write(self.textData.lawinfo['i2c'][int(d[1])])
                    wh.write('\t')
                    wh.write(self.textData.lawinfo['i2c'][int(d[2])])
                    wh.write('\n')
            wh.close()
        if args['model_arch'] in ['lstmiterib', 'lstmgrid']:
            return accuracy, exact_match/total, p,r,acc
        else:
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