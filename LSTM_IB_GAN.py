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
            nn.Linear(args['hiddenSize'], 1),
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

        self.embedding = LM.embedding

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

        self.attm = Parameter(torch.rand(args['hiddenSize'], args['hiddenSize'] * 2)).to(args['device'])

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
        self.encoderInputs = x['enc_input'].to(args['device'])
        self.encoder_lengths = x['enc_len']
        self.classifyLabels = x['labels'].to(args['device'])
        self.batch_size = self.encoderInputs.size()[0]
        self.seqlen = self.encoderInputs.size()[1]

        mask = torch.sign(self.encoderInputs).float()

        en_outputs, en_state = self.encoder_all(self.encoderInputs, self.encoder_lengths)  # batch seq hid

        en_hidden, en_cell = en_state  # 2 batch hid
        # print(en_hidden.size())
        en_hidden = en_hidden.transpose(0, 1)
        en_hidden = en_hidden.reshape(self.batch_size, args['hiddenSize'] * 2)
        att1 = torch.einsum('bsh,hg->bsg', en_outputs, self.attm)
        att2 = torch.einsum('bsg,bg->bs', att1, en_hidden)
        att2 = self.softmax(att2)
        z_nero_best = torch.einsum('bsh,bs->bh',en_outputs , att2)

        # z_nero_best = self.z_to_fea(en_outputs)
        # z_nero_best, _ = torch.max(z_nero_best, dim=1)  # batch hid
        # print(z_nero_best.size())
        output_all = self.ChargeClassifier(z_nero_best).to(args['device'])  # batch chargenum
        # print(output_all.size(), self.classifyLabels.size())
        recon_loss_all = self.NLLloss(output_all, self.classifyLabels).to(args['device'])
        recon_loss_mean_all = recon_loss_all #torch.mean(recon_loss_all, 1).to(args['device'])
        # try:
        en_outputs_select, en_state = self.encoder_select(self.encoderInputs, self.encoder_lengths)  # batch seq hid
        # except:
        #     print(self.encoderInputs, self.encoderInputs.size(), self.encoder_lengths)
        #     en_outputs_select, en_state = self.encoder_select(self.encoderInputs, self.encoder_lengths)  # batch seq hid
        # print(en_outputs.size())
        z_logit = self.x_2_prob_z(en_outputs_select.to(args['device']))  # batch seq 2

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
        # I_x_z = torch.mean(-torch.log(z_prob[:, :, 0] + eps), 1)
        I_x_z = (z_prob * torch.log(
            z_prob / torch.FloatTensor([0.9999, 0.0001]).unsqueeze(0).unsqueeze(1).to(args['device'])) + eps).sum(
            2).sum(1) * 0.01

        logp_z0 = torch.log(z_prob[:, :, 0])  # [B,T], log P(z = 0 | x)
        logp_z1 = torch.log(z_prob[:, :, 1])  # [B,T], log P(z = 1 | x)
        logpz = torch.where(sampled_seq[:, :, 1] == 0, logp_z0, logp_z1)
        logpz = mask * logpz

        # print(I_x_z)
        # en_hidden, en_cell = en_state   #2 batch hid
        # omega = torch.mean(torch.sum(torch.abs(sampled_seq[:,:-1,1] - sampled_seq[:,1:,1]), dim = 1))
        omega = self.LM.LMloss(sampled_seq_soft[:,:,1],sampled_seq[:, :, 1], self.encoderInputs)
        # print(I_x_z.size(), omega.size())
        # omega = torch.mean(omega, 1)

        output = self.ChargeClassifier(z_nero_sampled).to(args['device'])  # batch chargenum
        recon_loss = self.NLLloss(output, self.classifyLabels).to(args['device'])
        recon_loss_mean = recon_loss#  torch.mean(recon_loss, 1).to(args['device'])

        tt = torch.stack([recon_loss_mean.mean(), recon_loss_mean_all.mean(), I_x_z.mean(), omega.mean()])
        wordnum = torch.sum(mask, dim=1)
        sampled_num = torch.sum(sampled_seq[:,:,1], dim = 1) # batch
        sampled_num = (sampled_num == 0).float()  + sampled_num
        optional = {}
        optional["recon_loss"] = recon_loss_mean.mean().item()  # [1]
        optional['recon_best'] = recon_loss_mean_all.mean().item()
        optional['I_x_z'] = I_x_z.mean().item()
        optional['zdiff'] = omega.mean().item()

        return recon_loss_mean , recon_loss_mean_all , 0.0003 * I_x_z , 0.005*omega, z_nero_best, z_nero_sampled, output, sampled_seq, sampled_num/wordnum, logpz, optional


    def forward(self, x):
        losses,losses_best,I, om, z_nero_best, z_nero_sampled, _, _,_,logpz,optional = self.build(x)
        return losses,losses_best,I, om, z_nero_best, z_nero_sampled, logpz, optional

    def predict(self, x):
        _, _,_,_, _, _,output,sampled_words, wordsamplerate, _, _ = self.build(x)
        return output, (torch.argmax(output, dim=-1), sampled_words, wordsamplerate)


def train(textData, LM, model_path=args['rootDir'] + '/chargemodel_LSTM_IB_GAN.mdl', print_every=10000, plot_every=10,
          learning_rate=0.001, n_critic=5, eps = 1e-6):
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

    max_accu = -1
    print_every = 1000 if args['datasetsize'] == 'small' else print_every

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
            x['labels'] = autograd.Variable(torch.LongTensor(batch.label)).to(args['device'])
            if args['model_arch'] in ['lstmibgan', 'lstmibgan_law']:
                x['labels'] = x['labels'][:, 0]

            Gloss_pure,Gloss_best, I, om, z_nero_best, z_nero_sampled, logpz, optional = G_model(x)  # batch seq_len outsize
            Dloss = -torch.mean(torch.log(D_model(z_nero_best).clamp(eps,1))) + torch.mean(torch.log(D_model(z_nero_sampled.detach()).clamp(eps,1)))

            Dloss.backward(retain_graph=True)

            torch.nn.utils.clip_grad_norm_(D_model.parameters(), args['clip'])

            D_optimizer.step()

            # if i % n_critic == 0:
            # -----------------
            #  Train Generator
            # -----------------
            G_optimizer.zero_grad()
            # print(Gloss_pure.size() , D_model(z_nero_sampled).size() , logpz.size())
            G_ganloss = torch.log(D_model(z_nero_sampled).clamp(eps,1).squeeze())
            # Gloss = Gloss_best.mean() + Gloss_pure.mean() + 0.1*I.mean() + om.mean() - G_ganloss.mean() + ((0.01*Gloss_pure.detach() + I.detach() + om.detach()) * logpz.sum(1)).mean()
            # Gloss = Gloss_best.mean() + Gloss_pure.mean()+  (regu - torch.log(D_model(z_nero_sampled).squeeze()+eps) ).mean()
            if args['choose'] == 0:
                Gloss = 10 * Gloss_best.mean() + 10 * Gloss_pure.mean() + 80 * I.mean() + om.mean() - G_ganloss.mean() + (
                    (0.01 * Gloss_pure.detach() + I.detach() + om.detach()) * logpz.sum(1)).mean()
            elif args['choose'] == 1:
                Gloss = 10 * Gloss_pure.mean()
            elif args['choose'] == 2:
                Gloss = 10 * Gloss_pure.mean() + 80 * I.mean()
            elif args['choose'] == 3:
                Gloss = 10 * Gloss_best.mean() + 10 * Gloss_pure.mean() + 80 * I.mean() - G_ganloss.mean() + (
                    (0.01 * Gloss_pure.detach() + I.detach()) * logpz.sum(1)).mean()

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
                print(optional)

            if iter % plot_every == 0:
                plot_loss_avg = plot_Gloss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_Gloss_total = 0

            iter += 1
            # print(iter, datetime.datetime.now())

        accuracy, MP,MR, F = test(textData, G_model, 'test', max_accu)
        if accuracy > max_accu or max_accu == -1:
            print('accuracy = ', accuracy, '>= min_accuracy(', max_accu, '), saving model...')
            torch.save([G_model, D_model], model_path)
            max_accu = accuracy

        print('Epoch ', epoch, 'loss = ', sum(Glosses) / len(Glosses), 'Valid accuracy = ', accuracy,MP,MR, F , 'max accuracy=',
              max_accu)

    # self.test()
    # showPlot(plot_losses)


def test(textData, model, datasetname, max_accuracy):
    right = 0
    total = 0

    dset = []

    pppt = False
    TP_c = np.zeros(args['chargenum'])
    FP_c = np.zeros(args['chargenum'])
    FN_c = np.zeros(args['chargenum'])
    TN_c = np.zeros(args['chargenum'])
    with torch.no_grad():
        for batch in textData.getBatches(datasetname):
            x = {}
            x['enc_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs))
            x['enc_len'] = batch.encoder_lens
            x['labels'] = autograd.Variable(torch.LongTensor(batch.label)).to(args['device'])
            if args['model_arch'] in ['lstmibgan', 'lstmibgan_law']:
                x['labels'] = x['labels'][:, 0]

            output_probs, output_labels = model.predict(x)
            output_labels, sampled_words, wordsamplerate = output_labels
            if not pppt:
                pppt = True
                pind = np.random.choice(x['enc_input'].size()[0].item())
                for w, choice in zip(batch.encoderSeqs[pind], sampled_words[pind]):
                    if choice[1] == 1:
                        print('<', textData.index2word[w], '>', end='')
                    else:
                        print(textData.index2word[w], end='')

                print('sample rate: ', wordsamplerate[0])
            y = F.one_hot(torch.LongTensor(x['labels'].cpu().numpy()), num_classes=args['chargenum'])  # batch c
            y = y.bool().numpy()
            answer = output_labels.cpu().numpy()
            answer = F.one_hot(torch.LongTensor(answer), num_classes=args['chargenum'])  # batch c
            answer = answer.bool().numpy()

            tp_c = ((answer == True) & (answer == y)).sum(axis=0)  # c
            fp_c = ((answer == True) & (y == False)).sum(axis=0)  # c
            fn_c = ((answer == False) & (y == True)).sum(axis=0)  # c
            tn_c = ((answer == False) & (y == False)).sum(axis=0)  # c
            TP_c += tp_c
            FP_c += fp_c
            FN_c += fn_c
            TN_c += tn_c

            batch_correct = output_labels.cpu().numpy() == x['labels'].cpu().numpy()
            # print(output_labels.size(), torch.LongTensor(batch.label).size())
            right += sum(batch_correct)
            total += x['enc_input'].size()[0]

            for ind, c in enumerate(batch_correct):
                if not c:
                    dset.append((batch.encoderSeqs[ind], x['labels'][ind], output_labels[ind]))

    accuracy = right / total
    P_c = TP_c / (TP_c + FP_c)
    R_c = TP_c / (TP_c + FN_c)
    F_c = 2 * P_c * R_c / (P_c + R_c)
    F_macro = np.nanmean(F_c)
    MP = np.nanmean(P_c)
    MR = np.nanmean(R_c)

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

    return accuracy, MP, MR, F_macro