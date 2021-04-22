import functools
print = functools.partial(print, flush=True)
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter

import argparse
import numpy as np

import datetime, time
from Hyperparameters import args
from queue import PriorityQueue
import copy,math
from utils import *
from tqdm import tqdm
from textdataLM import TextData

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), datetime.datetime.now())


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
        self.batch_size = args['batchSize']

        self.dtype = 'float32'

        self.embedding = nn.Embedding(args['vocabularySize'], args['embeddingSize'])

        if args['decunit'] == 'lstm':
            self.dec_unit = nn.LSTM(input_size=args['embeddingSize'],
                                    hidden_size=args['hiddenSize'],
                                    num_layers=args['dec_numlayer'])
        elif args['decunit'] == 'gru':
            self.dec_unit = nn.GRU(input_size=args['embeddingSize'],
                                   hidden_size=args['hiddenSize'],
                                   num_layers=args['dec_numlayer'])

        self.out_unit = nn.Linear(args['hiddenSize'], args['vocabularySize'])
        self.logsoftmax = nn.LogSoftmax(dim = -1)

        self.element_len = args['hiddenSize']

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = -1)
        self.CEloss = torch.nn.CrossEntropyLoss(reduction='none')
        self.NLLloss = torch.nn.NLLLoss()
        self.sigmoid = torch.nn.Sigmoid()

        self.init_state = (torch.rand(args['dec_numlayer'], 1, args['hiddenSize']).to(args['device']),
                           torch.rand(args['dec_numlayer'], 1, args['hiddenSize']).to(args['device']))
        self.M = Parameter(torch.rand(args['hiddenSize'], args['embeddingSize']))

    def getloss(self, decoderInputs, decoderTargetsEmbedding, decoderTargets, inputmask = None, eps = 1e-20):
        '''
        :param decoderInputs:
        :param decoderTargetsEmbedding:  b s e
        :return:
        '''
        batch_size = decoderInputs.size()[0]
        dec_len = decoderInputs.size()[1]
        dec_input_embed = self.embedding(decoderInputs)
        # if mask is not None:
        #     dec_input_embed = dec_input_embed * mask.unsqueeze(2)

        init_state = (self.init_state[0].repeat([1, batch_size, 1]), self.init_state[1].repeat([1, batch_size, 1]))
        de_outputs, de_state = self.decoder_t(init_state, dec_input_embed, batch_size, dec_len) # b s h
        temp1 = torch.einsum('bsh,he->bse', de_outputs, self.M )# b s e
        temp2 = torch.einsum('bse,bse->bs', temp1, decoderTargetsEmbedding)
        probs = self.sigmoid(temp2)

        recon_loss = - torch.log(probs + eps)
        if inputmask is None:
            mask = torch.sign(decoderTargets.float())
            recon_loss = recon_loss * mask
        else:
            recon_loss = recon_loss * inputmask

        recon_loss_mean = torch.mean(recon_loss, dim = -1)
        return temp1, recon_loss_mean

    def buildModel(self, x, test = False, eps = 1e-20):
        self.decoderInputs = x['dec_input']
        self.decoder_lengths = x['dec_len']
        self.decoderTargets = x['dec_target']
        decoderTargetsEmbeddings = self.embedding(self.decoderTargets)
        temp1, recon_loss_mean = self.getloss(self.decoderInputs, decoderTargetsEmbeddings, self.decoderTargets)
        fake_recon_loss = 0

        if not test:
            negatives = self.get_negative_samples(self.decoderTargets)  # b s 10
            negativeEmbeddings = self.embedding(negatives)# b s 10 e
            fake_temp2 = torch.einsum('bse,bsne->bsn', temp1, negativeEmbeddings)
            fake_probs = self.sigmoid(fake_temp2)
            fake_recon_loss = - torch.log(fake_probs + eps)
            fake_recon_loss = torch.sum(fake_recon_loss, dim = 2) # b s
            fake_recon_loss = torch.sum(fake_recon_loss, dim = 1)
        return  recon_loss_mean, fake_recon_loss

    def get_negative_samples(self, indextensor, samplenum = 10):
        '''
        :param indextensor:  b s
        :return: b s num
        '''
        t1 = time.time()
        batch = indextensor.size()[0]
        seqlen = indextensor.size()[1]
        weights = torch.tensor([1 for _ in range(args['vocabularySize'])], dtype=torch.float)
        samples = torch.multinomial(weights, batch * seqlen * samplenum, replacement=True) # bs * 10
        res = samples.reshape((batch,seqlen, samplenum))
        # print(time.time() - t1)
        return res.to(args['device'])

    def decoder_t(self, initial_state, inputs, batch_size, dec_len):
        inputs = torch.transpose(inputs, 0,1).contiguous()
        state = initial_state

        output, out_state = self.dec_unit(inputs, state)

        # output = self.out_unit(output.view(batch_size * dec_len, args['hiddenSize']))
        # output = output.view(dec_len, batch_size, args['vocabularySize'])
        output = torch.transpose(output, 0,1)
        return output, out_state

    def forward(self, x, test = False):
        recon_loss_mean, fake_loss = self.buildModel(x, test)
        return  recon_loss_mean, fake_loss

    def LMloss(self, sampled_soft, sampled_hard, decoderInputs):

        decoderTargets = decoderInputs[:,1:]
        decoderInputs = decoderInputs[:,:-1]
        decoderTargetsEmbeddings = self.embedding(decoderTargets)
        decoderTargetsEmbeddings = decoderTargetsEmbeddings * sampled_hard[:,1:].unsqueeze(2)
        temp1, recon_loss_mean = self.getloss(decoderInputs, decoderTargetsEmbeddings, decoderTargets)
        return recon_loss_mean


def train(textData, model, model_path, print_every=10000, plot_every=10, learning_rate=0.001):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    print(type(textData.word2index))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-3, amsgrad=True)

    iter = 1
    batches = textData.getBatches()
    n_iters = len(batches)
    print('niters ', n_iters)

    args['trainseq2seq'] = False

    min_ppl = -1
    # accuracy = self.test('test', max_accu)
    for epoch in range(args['numEpochs']):
        losses = []

        for batch in tqdm(batches):
            optimizer.zero_grad()
            x = {}
            x['dec_input'] = autograd.Variable(torch.LongTensor(batch.decoderSeqs)).to(args['device'])
            x['dec_len'] = batch.decoder_lens
            x['dec_target'] = autograd.Variable(torch.LongTensor(batch.targetSeqs)).to(args['device'])

            loss , fake_loss = model(x)  # batch seq_len outsize
            loss = torch.mean(loss - fake_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['clip'])

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

        ppl = test(textData, model, 'test')
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
            x['dec_input'] = autograd.Variable(torch.LongTensor(batch.decoderSeqs)).to(args['device'])
            x['dec_len'] = batch.decoder_lens
            x['dec_target'] = autograd.Variable(torch.LongTensor(batch.targetSeqs)).to(args['device'])
            recon_loss_mean , _= model(x, test = True)  # batch seq_len outsize
            ave_loss = (ave_loss * num + sum(recon_loss_mean)) / (num + len(recon_loss_mean))
            num += len(recon_loss_mean)

    return torch.exp(ave_loss)

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g')
    parser.add_argument('--size', '-s')
    cmdargs = parser.parse_args()
    usegpu = True
    if cmdargs.gpu is None:
        usegpu = False
    else:
        usegpu = True
        args['device'] = 'cuda:' + str(cmdargs.gpu)

    if cmdargs.size is None:
        args['tasksize'] = 'big'
    else:
        args['tasksize'] = cmdargs.size

def kenlm_test(textData):
    import kenlm
    model = kenlm.Model('/Users/shalei/科研/2020/kenlm/law.lm')
    ppls = []
    for batch in textData.getBatches('test'):
        sentences = [' '.join([textData.index2word[wid] for wid in r if wid >3]) for r in batch.decoderSeqs]
        # print(sentences[0])
        ppl = [np.log(model.perplexity(s)) for s in sentences]
        # ppl = [-np.log(10**log_10_p) for log_10_p in ppl]
        ppls.extend(ppl)

    return sum(ppls)/len(ppls)

if __name__ == '__main__':
    parseargs()
    textData = TextData('cail')
    nll = kenlm_test(textData)
    print('LM=', nll, np.exp(nll))
    args['batchSize'] = 256
    # args['maxLength'] = 1000
    # args['maxLengthEnco'] = args['maxLength']
    # args['maxLengthDeco'] = args['maxLength'] + 1
    args['vocabularySize'] = textData.getVocabularySize()
    args['chargenum'] = textData.getChargeNum()
    model = LanguageModel(textData.word2index, textData.index2word).to(args['device'])
    train(textData, model, model_path = args['rootDir']+'/LM'+args['tasksize']+'.pkl')
