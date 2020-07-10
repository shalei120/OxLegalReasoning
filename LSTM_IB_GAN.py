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
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        print('Discriminator creation...')
        self.NLLloss = torch.nn.NLLLoss(reduction='none')
        self.disc = nn.Sequential(
            nn.Linear(args['hiddenSize'], args['hiddenSize']),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(args['hiddenSize'], 1),
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

    def __init__(self, w2i, i2w):
        """
        Args:
            args: parameters of the model
            textData: the dataset object
        """
        super(LSTM_IB_GAN_Model, self).__init__()
        print("Model creation...")

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

        self.ChargeClassifier = nn.Sequential(
            nn.Linear(args['hiddenSize'], args['chargenum']),
            nn.LogSoftmax(dim=-1)
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
        return y_hard

    def build(self, x, eps=0.000001):
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
        self.seqlen = self.encoderInputs.size()[1]

        mask = torch.sign(self.encoderInputs).float()

        en_outputs, en_state = self.encoder_all(self.encoderInputs, self.encoder_lengths)  # batch seq hid
        z_nero_best = self.z_to_fea(en_outputs)
        z_nero_best, _ = torch.max(z_nero_best, dim=1)  # batch hid
        output_all = self.ChargeClassifier(z_nero_best).to(args['device'])  # batch chargenum
        recon_loss_all = self.NLLloss(output_all, self.classifyLabels).to(args['device'])
        recon_loss_mean_all = torch.mean(recon_loss_all).to(args['device'])

        en_outputs_select, en_state = self.encoder_select(self.encoderInputs, self.encoder_lengths)  # batch seq hid
        # print(en_outputs.size())
        z_logit = self.x_2_prob_z(en_outputs_select.to(args['device']))  # batch seq 2

        z_logit_fla = z_logit.reshape((self.batch_size * self.seqlen, 2))
        sampled_seq = self.gumbel_softmax(z_logit_fla).reshape((self.batch_size, self.seqlen, 2))  # batch seq  //0-1
        sampled_seq = sampled_seq * mask.unsqueeze(2)

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

        output = self.ChargeClassifier(z_nero_sampled).to(args['device'])  # batch chargenum
        recon_loss = self.NLLloss(output, self.classifyLabels).to(args['device'])
        recon_loss_mean = torch.mean(recon_loss).to(args['device'])

        tt = torch.stack([recon_loss_mean, recon_loss_mean_all, I_x_z])

        return recon_loss_mean + recon_loss_mean_all + 0.01 * I_x_z, z_nero_best, z_nero_sampled, output, tt

    def forward(self, x):
        losses, z_nero_best, z_nero_sampled, _, tt = self.build(x)
        return losses, z_nero_best, z_nero_sampled, tt

    def predict(self, x):
        _, _, _, output, _ = self.build(x)
        return output, torch.argmax(output, dim=-1)


def train(textData, model_path=args['rootDir'] + '/chargemodel_LSTM_IB_GAN.mdl', print_every=10000, plot_every=10,
          learning_rate=0.001, n_critic=5):
    start = time.time()
    plot_losses = []
    print_Gloss_total = 0  # Reset every print_every
    plot_Gloss_total = 0  # Reset every plot_every
    print_Dloss_total = 0  # Reset every print_every
    plot_Dloss_total = 0  # Reset every plot_every
    G_model = LSTM_IB_GAN_Model(textData.word2index, textData.index2word).to(args['device'])
    D_model = Discriminator().to(args['device'])

    print(type(textData.word2index))

    G_optimizer = optim.Adam(G_model.parameters(), lr=learning_rate, eps=1e-3, amsgrad=True)
    D_optimizer = optim.Adam(D_model.parameters(), lr=learning_rate, eps=1e-3, amsgrad=True)

    iter = 1
    batches = textData.getBatches()
    n_iters = len(batches)
    print('niters ', n_iters)

    args['trainseq2seq'] = False

    max_accu = -1

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

            for ind in range(index, index+5):
                ind = ind % n_iters
                D_optimizer.zero_grad()
                x = {}
                x['enc_input'] = autograd.Variable(torch.LongTensor(batches[ind].encoderSeqs)).to(args['device'])
                x['enc_len'] = batches[ind].encoder_lens
                x['labels'] = autograd.Variable(torch.LongTensor(batches[ind].label)).to(args['device'])
                x['labels'] = x['labels'][:, 0]

                Gloss_pure, z_nero_best, z_nero_sampled, tt = G_model(x)  # batch seq_len outsize
                Dloss = -torch.mean(D_model(z_nero_best)) + torch.mean(D_model(z_nero_sampled))

                Dloss.backward(retain_graph=True)

                torch.nn.utils.clip_grad_norm_(D_model.parameters(), args['clip'])

                D_optimizer.step()

            # if i % n_critic == 0:
            # -----------------
            #  Train Generator
            # -----------------
            # for param in G_model.parameters():
            #     param.requires_grad = True
            # for param in D_model.parameters():
            #     param.requires_grad = False

            x = {}
            x['enc_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs)).to(args['device'])
            x['enc_len'] = batch.encoder_lens
            x['labels'] = autograd.Variable(torch.LongTensor(batch.label)).to(args['device'])
            x['labels'] = x['labels'][:, 0]

            G_optimizer.zero_grad()
            Gloss_pure, z_nero_best, z_nero_sampled, tt = G_model(x)  # batch seq_len outsize
            Gloss = Gloss_pure - torch.mean(D_model(z_nero_sampled))
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

        accuracy = test(textData, G_model, 'test', max_accu)
        if accuracy > max_accu or max_accu == -1:
            print('accuracy = ', accuracy, '>= min_accuracy(', max_accu, '), saving model...')
            torch.save([G_model, D_model], model_path)
            max_accu = accuracy

        print('Epoch ', epoch, 'loss = ', sum(Glosses) / len(Glosses), 'Valid accuracy = ', accuracy, 'max accuracy=',
              max_accu)

    # self.test()
    # showPlot(plot_losses)


def test(textData, model, datasetname, max_accuracy):
    right = 0
    total = 0

    dset = []

    with torch.no_grad():
        for batch in textData.getBatches(datasetname):
            x = {}
            x['enc_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs))
            x['enc_len'] = batch.encoder_lens
            x['labels'] = autograd.Variable(torch.LongTensor(batch.label)).to(args['device'])
            x['labels'] = x['labels'][:, 0]

            output_probs, output_labels = model.predict(x)

            batch_correct = output_labels.cpu().numpy() == x['labels'].cpu().numpy()
            # print(output_labels.size(), torch.LongTensor(batch.label).size())
            right += sum(batch_correct)
            total += x['enc_input'].size()[0]

            for ind, c in enumerate(batch_correct):
                if not c:
                    dset.append((batch.encoderSeqs[ind], x['labels'][ind], output_labels[ind]))

    accuracy = right / total

    # if accuracy > max_accuracy:
    #     with open(args['rootDir'] + '/error_case_' + args['model_arch'] + '.txt', 'w') as wh:
    #         for d in dset:
    #             wh.write(''.join([textData.index2word[wid] for wid in d[0]]))
    #             wh.write('\t')
    #             wh.write(textData.lawinfo['i2c'][int(d[1])])
    #             wh.write('\t')
    #             wh.write(textData.lawinfo['i2c'][int(d[2])])
    #             wh.write('\n')
    #     wh.close()

    return accuracy