import functools
print = functools.partial(print, flush=True)
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter

import numpy as np
import datetime
import time, math
from Encoder import Encoder
from Decoder import Decoder
from Hyperparameters import args


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


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        print('Discriminator creation...')
        self.NLLloss = torch.nn.NLLLoss(reduction='none')
        self.disc = nn.Sequential(
            nn.Linear(args['hiddenSize'], args['hiddenSize']),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Sigmoid(),
        ).to(args['device'])

    def forward(self, z_nero):
        '''
        :param unsampled_word: batch seq hid
        :return:
        '''

        G_judge_score = self.disc(z_nero)
        return G_judge_score


class LSTM_IB_GAN_Model(nn.Module):
    """
    Implementation of a seq2seq model.
    Architecture:
        Encoder/decoder
        2 LTSM layers
    """

    def __init__(self, w2i, i2w, LM):
        """
        Args:
            args: parameters of the model
            textData: the dataset object
        """
        super(LSTM_IB_GAN_Model, self).__init__()
        print("Model creation...")

        self.LM = LM
        self.word2index = w2i
        self.index2word = i2w
        self.max_length = args['maxLengthDeco']

        self.NLLloss = torch.nn.NLLLoss(reduction='none')
        self.CEloss = torch.nn.CrossEntropyLoss(reduction='none')

        self.embedding = nn.Embedding(args['vocabularySize'], args['embeddingSize']).to(args['device'])

        self.encoder_all = Encoder(w2i, i2w, self.embedding).to(args['device'])
        self.encoder_select = Encoder(w2i, i2w, self.embedding, bidirectional = True).to(args['device'])
        self.encoder_mask = Encoder(w2i, i2w, self.embedding).to(args['device'])

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        self.x_2_prob_z = nn.Sequential(
            nn.Linear(args['hiddenSize'] * 2, 2)
        ).to(args['device'])
        self.z_to_fea = nn.Linear(args['hiddenSize'], args['hiddenSize']).to(args['device'])

        self.review_scorer = nn.Sequential(
            nn.Linear(args['hiddenSize'], args['chargenum']),
            nn.Sigmoid()
        ).to(args['device'])

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).to(args['device'])
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature=args['temperature']):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y
        return y_hard, y

    def build(self, x, eps=0.000001):
        '''
        :param encoderInputs: [batch, enc_len]
        :param decoderInputs: [batch, dec_len]
        :param decoderTargets: [batch, dec_len]
        :return:
        '''

        # print(x['enc_input'])
        self.encoderInputs = x['enc_input']
        self.encoder_lengths = x['enc_len']
        self.scores = x['labels']
        self.batch_size = self.encoderInputs.size()[0]
        self.seqlen = self.encoderInputs.size()[1]

        mask = torch.sign(self.encoderInputs).float()

        en_outputs, en_state = self.encoder_all(self.encoderInputs, self.encoder_lengths)  # batch seq hid
        z_nero_best = self.z_to_fea(en_outputs)
        z_nero_best, _ = torch.max(z_nero_best, dim=1)  # batch hid
        output_all = self.review_scorer(z_nero_best)  # batch 5
        recon_loss_all = (output_all - self.scores) ** 2
        recon_loss_mean_all = torch.mean(recon_loss_all[:, args['aspect']])

        en_outputs_select, en_state = self.encoder_select(self.encoderInputs, self.encoder_lengths)  # batch seq hid
        # print(en_outputs.size())
        z_logit = self.x_2_prob_z(en_outputs_select)  # batch seq 2

        z_logit_fla = z_logit.reshape((self.batch_size * self.seqlen, 2))
        sampled_seq, sampled_seq_soft = self.gumbel_softmax(z_logit_fla) # batch seq  //0-1
        sampled_seq = sampled_seq.reshape((self.batch_size, self.seqlen, 2))
        sampled_seq_soft = sampled_seq_soft.reshape((self.batch_size, self.seqlen, 2))
        sampled_seq = sampled_seq * mask.unsqueeze(2)
        sampled_seq_soft = sampled_seq_soft * mask.unsqueeze(2)
        # print(sampled_seq)

        # sampled_word = self.encoderInputs * (sampled_seq[:,:,1])  # batch seq

        en_outputs_masked, en_state = self.encoder_mask(self.encoderInputs, self.encoder_lengths,
                                                        sampled_seq[:, :, 1])  # batch seq hid

        s_w_feature = self.z_to_fea(en_outputs_masked)
        z_nero_sampled, _ = torch.max(s_w_feature, dim=1)  # batch hid

        z_prob = self.softmax(z_logit)
        I_x_z = torch.mean(-torch.log(z_prob[:, :, 0] + eps))
        # print(I_x_z)
        # en_hidden, en_cell = en_state   #2 batch hid
        # omega = torch.mean(torch.sum(torch.abs(sampled_seq[:,:-1,1] - sampled_seq[:,1:,1]), dim = 1))
        omega = self.LM.LMloss(sampled_seq_soft[:,:,1],sampled_seq[:, :, 1], self.encoderInputs)
        omega = torch.mean(omega)

        output = self.review_scorer(z_nero_sampled)  # batch aspectnum
        recon_loss = (output - self.scores)**2
        recon_loss_mean = torch.mean(recon_loss[:, args['aspect']])

        tt = torch.stack([recon_loss_mean, recon_loss_mean_all, I_x_z, omega])
        wordnum = torch.sum(mask, dim=1)
        sampled_num = torch.sum(sampled_seq[:,:,1], dim = 1) # batch
        sampled_num = (sampled_num == 0).float()  + sampled_num

        return recon_loss_mean + recon_loss_mean_all + 0.05 * I_x_z + 0.001*omega, z_nero_best, z_nero_sampled, output, sampled_seq, sampled_num/wordnum, tt


    def forward(self, x):
        losses, z_nero_best, z_nero_sampled, _, _,_,tt = self.build(x)
        return losses, z_nero_best, z_nero_sampled, tt

    def predict(self, x):
        _, _, _, output,sampled_words, wordsamplerate, _ = self.build(x)
        return output, ( sampled_words, wordsamplerate)


def train(textData, LM, model_path=args['rootDir'] + '/chargemodel_LSTM_IB_GAN.mdl', print_every=10000, plot_every=10,
          learning_rate=0.001, n_critic=5):
    start = time.time()
    plot_losses = []
    print_Gloss_total = 0  # Reset every print_every
    plot_Gloss_total = 0  # Reset every plot_every
    print_Dloss_total = 0  # Reset every print_every
    plot_Dloss_total = 0  # Reset every plot_every
    G_model = LSTM_IB_GAN_Model(textData.word2index, textData.index2word, LM).to(args['device'])
    D_model = Discriminator().to(args['device'])

    print(type(textData.word2index))

    G_optimizer = optim.Adam(G_model.parameters(), lr=learning_rate, eps=1e-3, amsgrad=True)
    D_optimizer = optim.Adam(D_model.parameters(), lr=learning_rate, eps=1e-3, amsgrad=True)

    iter = 1
    batches = textData.getBatches()
    n_iters = len(batches)
    print('niters ', n_iters)

    args['trainseq2seq'] = False

    max_p = -1

    # accuracy = test(textData, G_model, 'test', max_accu)
    for epoch in range(args['numEpochs']):
        Glosses = []
        Dlosses = []

        for index, batch in enumerate(batches):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            # for param in G_model.parameters():
            #     param.requires_grad = False
            # for param in D_model.parameters():
            #     param.requires_grad = True

            # for ind in range(index, index+5):
            #     ind = ind % n_iters
            D_optimizer.zero_grad()
            x = {}
            x['enc_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs)).to(args['device'])
            x['enc_len'] = batch.encoder_lens
            x['labels'] = autograd.Variable(torch.FloatTensor(batch.label)).to(args['device'])


            Gloss_pure, z_nero_best, z_nero_sampled, tt = G_model(x)  # batch seq_len outsize
            Dloss = -torch.mean(torch.log(D_model(z_nero_best))) + torch.mean(torch.log(D_model(z_nero_sampled.detach())))

            Dloss.backward(retain_graph=True)

            torch.nn.utils.clip_grad_norm_(D_model.parameters(), args['clip'])

            D_optimizer.step()

            # if i % n_critic == 0:
            # -----------------
            #  Train Generator
            # -----------------
            G_optimizer.zero_grad()
            Gloss = Gloss_pure - torch.mean(torch.log(D_model(z_nero_sampled)))
            Gloss.backward(retain_graph=True)
            G_optimizer.step()

            print_Gloss_total += Gloss.data
            plot_Gloss_total += Gloss.data

            print_Dloss_total += Dloss.data
            plot_Dloss_total += Dloss.data

            Glosses.append(Gloss.data)
            Dlosses.append(Dloss.data)

            if iter % print_every == 0:
                print_Gloss_avg = print_Gloss_total / print_every
                print_Gloss_total = 0
                print_Dloss_avg = print_Dloss_total / print_every
                print_Dloss_total = 0
                print('%s (%d %d%%) %.4f %.4f' % (timeSince(start, iter / (n_iters * args['numEpochs'])),
                                                  iter, iter / n_iters * 100, print_Gloss_avg, print_Dloss_avg), end='')
                print(tt)

            if iter % plot_every == 0:
                plot_loss_avg = plot_Gloss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_Gloss_total = 0

            iter += 1
            # print(iter, datetime.datetime.now())

        MSEloss, total_prec, samplerate = test(textData, G_model, 'test', max_accu)
        if total_prec > max_p or max_p == -1:
            print('total_prec = ', total_prec, '>= max_p(', max_p, '), saving model...')
            torch.save([G_model, D_model], model_path)
            max_p = total_prec

        print('Epoch ', epoch, 'loss = ', sum(Glosses) / len(Glosses), 'Valid  = ', MSEloss, total_prec, samplerate, 'max prec=',
              max_p)

    # self.test()
    # showPlot(plot_losses)


def test(textData, model, datasetname, max_accuracy):
    total = 0

    dset = []

    pppt = False
    MSEloss = 0
    total_prec = 0.0
    samplerate = 0.0
    with torch.no_grad():
        for batch in textData.getBatches(datasetname):
            x = {}
            x['enc_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs)).to(args['device'])
            x['enc_len'] = batch.encoder_lens
            x['labels'] = autograd.Variable(torch.FloatTensor(batch.label)).to(args['device'])

            output_probs, output_labels = model.predict(x)
            sampled_words, wordsamplerate = output_labels
            if not pppt:
                pppt = True
                for w, choice in zip(batch.encoderSeqs[0], sampled_words[0]):
                    if choice[1] == 1:
                        print('<',textData.index2word[w],'>', end='')
                    else:
                        print(textData.index2word[w], end='')

                print('sample rate: ', wordsamplerate[0])

            batch_correct = (output_probs.cpu().numpy() - x['labels'].cpu().numpy())**2
            batch_correct = batch_correct[:, args['aspect']]
            # print(output_labels.size(), torch.LongTensor(batch.label).size())
            # right += sum(batch_correct[:, args['aspect']])
            MSEloss = (MSEloss * total + batch_correct.sum(dim = 0) ) / (total + x['enc_input'].size()[0])

            seqlen = torch.sign(x['enc_input']).float().sum(dim = 1) # batch
            prec = 0.0
            for i, b in enumerate(batch):
                right = 0
                for w, choice in zip(batch.encoderSeqs[i], sampled_words[i]):
                    if choice[1] == 1 and w in batch.rationals[i][args['aspect']]:
                        right += 1
                prec += right / seqlen[i]
            total_prec =( total_prec * total + prec) / (total + x['enc_input'].size()[0])
            samplerate = (samplerate * total + wordsamplerate.sum(dim = 0))/ (total + x['enc_input'].size()[0])

            total += x['enc_input'].size()[0]


            # for ind, c in enumerate(batch_correct):
            #     if not c:
            #         dset.append((batch.encoderSeqs[ind], x['labels'][ind], output_labels[ind]))


    return MSEloss, total_prec, samplerate