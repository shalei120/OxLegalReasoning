import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter

import numpy as np

import datetime


from Encoder import Encoder
from Decoder import Decoder
from Hyperparameters import args

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        print('Discriminator creation...')
        self.NLLloss = torch.nn.NLLLoss(reduction = 'none')
        self.z_to_fea = nn.Linear(args['hiddenSize'], args['hiddenSize']).to(args['device'])
        self.ChargeClassifier = nn.Sequential(
            nn.Linear(args['hiddenSize'], args['chargenum']),
            nn.LogSoftmax(dim=-1)
        ).to(args['device'])

    def forward(self, unsampled_word, labels):
        '''
        :param unsampled_word: batch seq hid
        :return:
        '''
        s_w_feature = self.z_to_fea(unsampled_word)
        s_w_feature, _ = torch.max(s_w_feature, dim = 1) # batch hid
        output = self.ChargeClassifier(s_w_feature).to(args['device'])  # batch chargenum
        recon_loss = self.NLLloss(output, labels).to(args['device'])
        recon_loss_mean = torch.mean(recon_loss).to(args['device'])
        max_prob, _ = torch.max(output, dim = -1) # batch
        return recon_loss_mean, torch.mean(max_prob)

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

        self.NLLloss = torch.nn.NLLLoss(reduction = 'none')
        self.CEloss =  torch.nn.CrossEntropyLoss(reduction = 'none')

        self.embedding = nn.Embedding(args['vocabularySize'], args['embeddingSize']).to(args['device'])

        self.encoder = Encoder(w2i, i2w, self.embedding).to(args['device'])

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = -1)

        self.x_2_prob_z = nn.Sequential(
            nn.Linear(args['hiddenSize'], 2),
            nn.Softmax(dim=-1)
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

    def gumbel_softmax(self, logits, temperature = args['temperature']):
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

    def forward(self, x, eps = 0.000001):
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

        en_outputs, en_state = self.encoder(self.encoderInputs, self.encoder_lengths)  # batch seq hid
        # print(en_outputs.size())
        z_prob = self.x_2_prob_z(en_outputs.to(args['device'])) # batch seq 2

        z_prob_fla = z_prob.reshape((self.batch_size * self.seqlen, 2))
        sampled_seq = self.gumbel_softmax(z_prob_fla).reshape((self.batch_size, self.seqlen, 2))  # batch seq  //0-1
        sampled_seq = sampled_seq * mask.unsqueeze(2)

        # print(sampled_seq)

        sampled_word = en_outputs * (sampled_seq[:,:,1].unsqueeze(2))  # batch seq hid
        s_w_feature = self.z_to_fea(sampled_word)
        s_w_feature, _ = torch.max(s_w_feature, dim = 1) # batch hid

        unsampled_word = en_outputs * (sampled_seq[:,:,0].unsqueeze(2))  # batch seq hid

        I_x_z = torch.mean(-torch.log(z_prob[:,:,0]+ eps))
        # print(I_x_z)
        # en_hidden, en_cell = en_state   #2 batch hid

        output = self.ChargeClassifier(s_w_feature).to(args['device'])  # batch chargenum

        recon_loss = self.NLLloss(output, self.classifyLabels).to(args['device'])

        recon_loss_mean = torch.mean(recon_loss).to(args['device'])

        return recon_loss_mean + I_x_z, unsampled_word

    def predict(self, x):
        encoderInputs = x['enc_input'].to(args['device'])
        encoder_lengths = x['enc_len']

        batch_size = encoderInputs.size()[0]
        seqlen = encoderInputs.size()[1]
        mask = torch.sign(encoderInputs).float()

        en_outputs, en_state = self.encoder(encoderInputs, encoder_lengths)

        z_prob = self.x_2_prob_z(en_outputs.to(args['device']))  # batch seq 2

        z_prob_fla = z_prob.reshape((batch_size * seqlen, 2))
        sampled_seq = self.gumbel_softmax(z_prob_fla).reshape((batch_size, seqlen, 2))  # batch seq  //0-1
        sampled_seq = sampled_seq * mask.unsqueeze(2)

        sampled_word = en_outputs * (sampled_seq[:, :, 1].unsqueeze(2))  # batch seq hid
        s_w_feature = self.z_to_fea(sampled_word)
        s_w_feature, _ = torch.max(s_w_feature, dim=1)  # batch hid


        output = self.ChargeClassifier(s_w_feature).to(args['device'])  # batch chargenum


        return output, torch.argmax(output, dim = -1)

def train(textData, model_path = args['rootDir'] + '/chargemodel_LSTM_IB_GAN.mdl', print_every=10000, plot_every=10, learning_rate=0.001):
    start = time.time()
    plot_losses = []
    print_Gloss_total = 0  # Reset every print_every
    plot_Gloss_total = 0  # Reset every plot_every
    print_Dloss_total = 0  # Reset every print_every
    plot_Dloss_total = 0  # Reset every plot_every
    G_model = LSTM_IB_GAN_Model(textData.word2index, textData.index2word)
    D_model = Discriminator()

    print(type(textData.word2index))

    G_optimizer = optim.Adam(G_model.parameters(), lr=learning_rate, eps=1e-3, amsgrad=True)
    D_optimizer = optim.Adam(D_model.parameters(), lr=learning_rate, eps=1e-3, amsgrad=True)

    iter = 1
    batches = textData.getBatches()
    n_iters = len(batches)
    print('niters ', n_iters)

    args['trainseq2seq'] = False

    max_accu = -1

    for epoch in range(args['numEpochs']):
        Glosses = []
        Dlosses = []

        for batch in batches:

            # ---------------------
            #  Train Discriminator
            # ---------------------

            D_optimizer.zero_grad()
            x = {}
            x['enc_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs)).to(args['device'])
            x['enc_len'] = batch.encoder_lens
            x['labels'] = autograd.Variable(torch.LongTensor(batch.label)).to(args['device'])

            Gloss_pure, unsampled_words = G_model(x)  # batch seq_len outsize
            Dloss, G_judge_score = D_model(unsampled_words, x['labels'])


            Dloss.backward(retain_graph=True)

            torch.nn.utils.clip_grad_norm_(D_model.parameters(), args['clip'])

            D_optimizer.step()

            # -----------------
            #  Train Generator
            # -----------------
            G_optimizer.zero_grad()

            Gloss = Gloss_pure + G_judge_score
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
                                             iter, iter / n_iters * 100, print_Gloss_avg, print_Dloss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_Gloss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_Gloss_total = 0

            iter += 1

        accuracy = test(textData, G_model, 'test', max_accu)
        if accuracy > max_accu or max_accu == -1:
            print('accuracy = ', accuracy, '>= min_accuracy(', max_accu, '), saving model...')
            torch.save([G_model, D_model], model_path)
            max_accu = accuracy

        print('Epoch ', epoch, 'loss = ', sum(losses) / len(losses), 'Valid accuracy = ', accuracy, 'max accuracy=', max_accu)

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

            output_probs, output_labels = model.predict(x)

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
                wh.write(''.join([textData.index2word[wid] for wid in d[0]]))
                wh.write('\t')
                wh.write(textData.lawinfo['i2c'][int(d[1])])
                wh.write('\t')
                wh.write(textData.lawinfo['i2c'][int(d[2])])
                wh.write('\n')
        wh.close()

    return accuracy