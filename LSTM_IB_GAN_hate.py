import functools
print = functools.partial(print, flush=True)
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

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

class MulModule(nn.Module):
    def __init__(self, time):
        super(MulModule, self).__init__()
        self.time = time

    def forward(self, x) :
        return x * self.time

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        print('Discriminator creation...')
        self.NLLloss = torch.nn.NLLLoss(reduction='none')
        self.disc = nn.Sequential(
            nn.Linear(args['hiddenSize']*2, 1),
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

    def __init__(self, w2i, i2w, LM=None):
        """
        Args:
            args: parameters of the model
            textData: the dataset object
        """
        super(LSTM_IB_GAN_Model, self).__init__()
        print("Model creation...")

        # self.LM = LM
        self.word2index = w2i
        self.index2word = i2w
        self.max_length = args['maxLengthDeco']

        self.NLLloss = torch.nn.NLLLoss(reduction='none')
        self.CEloss = torch.nn.CrossEntropyLoss(reduction='none')

        self.embedding = nn.Embedding(args['vocabularySize'], args['embeddingSize'])
            #LM.embedding

        self.encoder_all = Encoder(w2i, i2w, self.embedding, bidirectional = True).to(args['device'])
        self.encoder_select = Encoder(w2i, i2w, self.embedding, bidirectional = True).to(args['device'])
        self.encoder_mask = Encoder(w2i, i2w, self.embedding, bidirectional = True).to(args['device'])

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.drop_layer = nn.Dropout(p=0.2)

        self.x_2_prob_z = nn.Sequential(
            nn.Linear(args['hiddenSize'] * 2, 2)
        ).to(args['device'])
        self.z_to_fea = nn.Linear(args['hiddenSize']*2, args['hiddenSize']*2).to(args['device'])

        self.ChargeClassifier = nn.Sequential(
            nn.Linear(args['hiddenSize']*2, 3),
            # MulModule(50),
            nn.LogSoftmax(dim=-1)
        ).to(args['device'])

        self.attm = Parameter(torch.rand(args['hiddenSize']*2, args['hiddenSize'] * 4)).to(args['device'])

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
        en_hidden = en_hidden.reshape(self.batch_size, args['hiddenSize'] * 4)
        att1 = torch.einsum('bsh,hg->bsg', en_outputs, self.attm)
        att2 = torch.einsum('bsg,bg->bs', att1, en_hidden)
        att2 = self.softmax(att2)
        z_nero_best = torch.einsum('bsh,bs->bh',en_outputs , att2)
        # z_nero_best = self.drop_layer(z_nero_best)

        # z_nero_best = self.z_to_fea(en_outputs)
        # z_nero_best, _ = torch.max(z_nero_best, dim=1)  # batch hid
        # print(z_nero_best.size())
        output_all = self.ChargeClassifier(z_nero_best).to(args['device'])  # batch chargenum
        recon_loss_all = self.NLLloss(output_all, self.classifyLabels).to(args['device'])
        recon_loss_mean_all = recon_loss_all #torch.mean(recon_loss_all, 1).to(args['device'])
        # try:
        en_outputs_select, en_state = self.encoder_select(self.encoderInputs, self.encoder_lengths)  # batch seq hid
        # except:
        #     print(self.encoderInputs, self.encoderInputs.size(), self.encoder_lengths)
        #     en_outputs_select, en_state = self.encoder_select(self.encoderInputs, self.encoder_lengths)  # batch seq hid
        # print(en_outputs.size())
        z_logit = self.x_2_prob_z(en_outputs_select.to(args['device']))  # batch seq 2
        self.z_prob = z_prob = self.softmax(z_logit) # batch seq 2

        if args['training']:
            z_logit_fla = z_logit.reshape((self.batch_size * self.seqlen, 2))
            sampled_seq, sampled_seq_soft = self.gumbel_softmax(z_logit_fla) # batch seq  //0-1
            sampled_seq = sampled_seq.reshape((self.batch_size, self.seqlen, 2))
            sampled_seq_soft = sampled_seq_soft.reshape((self.batch_size, self.seqlen, 2))
            sampled_seq = sampled_seq * mask.unsqueeze(2)
            sampled_seq_soft = sampled_seq_soft * mask.unsqueeze(2)
        else:
            # print('Evaluating...')
            # print()
            sampled_seq = ((z_prob > 0.5)*mask.unsqueeze(2)).float()
            # _, rationale_indexes = torch.topk(z_prob[:,:,1]*mask, 5)
            # rationale_onehot = F.one_hot(rationale_indexes, num_classes = self.seqlen )
            # rationale_onehot = rationale_onehot.sum(1)
            # sampled_seq = torch.cat([1-rationale_onehot.unsqueeze(2), rationale_onehot.unsqueeze(2)], dim = 2).float()
        # print(sampled_seq)

        # sampled_word = self.encoderInputs * (sampled_seq[:,:,1])  # batch seq

        en_outputs_masked, en_state = self.encoder_mask(self.encoderInputs, self.encoder_lengths,
                                                        sampled_seq[:, :, 1])  # batch seq hid

        s_w_feature = self.z_to_fea(en_outputs_masked)
        # s_w_feature = self.drop_layer(s_w_feature)
        z_nero_sampled, _ = torch.max(s_w_feature, dim=1)  # batch hid
        z_nero_sampled = self.drop_layer(z_nero_sampled)

        # I_x_z = torch.mean(-torch.log(z_prob[:, :, 0] + eps), 1)
        I_x_z = (mask.unsqueeze(2)*z_prob * torch.log(z_prob / torch.FloatTensor([0.9999,0.0001]).unsqueeze(0).unsqueeze(1).to(args['device']))+eps).sum(2).sum(1) * 0.01

        logp_z0 = torch.log(z_prob[:, :, 0])  # [B,T], log P(z = 0 | x)
        logp_z1 = torch.log(z_prob[:, :, 1])  # [B,T], log P(z = 1 | x)
        logpz = torch.where(sampled_seq[:, :, 1] == 0, logp_z0, logp_z1)
        logpz = mask * logpz

        # print(I_x_z)
        # en_hidden, en_cell = en_state   #2 batch hid
        omega = torch.mean(torch.sum(torch.abs(sampled_seq[:,:-1,1] - sampled_seq[:,1:,1]), dim = 1))
        # omega = self.LM.LMloss(sampled_seq_soft[:,:,1],sampled_seq[:, :, 1], self.encoderInputs)
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

        # m(x/r)

        en_outputs_norational, _ = self.encoder_mask(self.encoderInputs, self.encoder_lengths,
                                                        sampled_seq[:, :, 0])  # batch seq hid
        s_w_feature_norational = self.z_to_fea(en_outputs_norational.detach())
        z_nero_norational, _ = torch.max(s_w_feature_norational.detach(), dim=1)  # batch hid
        z_nero_norational = z_nero_norational.detach()*0
        output_norational = self.ChargeClassifier(z_nero_norational.detach()).to(args['device'])  # batch chargenum

        return recon_loss_mean , recon_loss_mean_all , 0.0003 * I_x_z , 0.005*omega, z_nero_best, z_nero_sampled, output, sampled_seq, sampled_num/wordnum, logpz, optional, z_prob[:,:,1], output_all, output_norational


    def forward(self, x):
        losses,losses_best,I, om, z_nero_best, z_nero_sampled, _, _,_,logpz,optional , _, _, _= self.build(x)
        return losses,losses_best,I, om, z_nero_best, z_nero_sampled, logpz, optional

    def predict(self, x):
        _, _,_,_, _, _,output,sampled_words, wordsamplerate, _, _, prob_rational, output_all , output_norational= self.build(x)
        return torch.exp(output), (torch.argmax(output, dim=-1), sampled_words, wordsamplerate, prob_rational, self.softmax(50*torch.exp(output_all)), torch.exp(output_norational))


def train(textData, LM=None, model_path=args['rootDir'] + '/chargemodel_LSTM_IB_GAN_small.mdl', print_every=500, plot_every=10,
          learning_rate=0.001, n_critic=5, eps = 1e-6):
    args['training'] = True
    print('Using small arch...')
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

    accuracy = test(textData, G_model, 'test', max_accu)
    max_res = {}
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
            # if args['model_arch'] in ['lstmibgan', 'lstmibgan_law']:
            #     x['labels'] = x['labels'][:, 0]

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
            if args['choose'] == 0:
                Gloss = 10*Gloss_best.mean() + 10*Gloss_pure.mean() + 1000*I.mean() + om.mean() - G_ganloss.mean() #+ ((0.01*Gloss_pure.detach() + I.detach() + om.detach()) * logpz.sum(1)).mean()
            elif args['choose'] == 1:
                Gloss = 10 * Gloss_best.mean() + 10 * Gloss_pure.mean() + 80 * I.mean() + om.mean() - G_ganloss.mean() + (
                            (0.01 * Gloss_pure.detach() + I.detach() + om.detach()) * logpz.sum(1)).mean()
            elif args['choose'] == 2:
                Gloss = Gloss_best.mean() + Gloss_pure.mean() + 35 * I.mean() + 0 * om.mean() - G_ganloss.mean()  # + ((0.01*Gloss_pure.detach() + I.detach() + om.detach()) * logpz.sum(1)).mean()
            elif args['choose'] == 3:
                Gloss =  Gloss_best.mean() + Gloss_pure.mean() + 35*I.mean() + 1*om.mean() - G_ganloss.mean()  + ((0.01*Gloss_pure.detach()+ I.detach() + om.detach()) * logpz.sum(1)).mean()
            elif args['choose'] == 4:
                Gloss = Gloss_pure.mean()
            elif args['choose'] == 5:
                Gloss = 10 * Gloss_pure.mean() + 80 * I.mean() + om.mean() + (
                            (0.01 * Gloss_pure.detach() + I.detach() + om.detach()) * logpz.sum(1)).mean()
            # Gloss = Gloss_best.mean() + Gloss_pure.mean()+  (regu - torch.log(D_model(z_nero_sampled).squeeze()+eps) ).mean()
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

        res = test(textData, G_model, 'test', max_accu)
        # if res['accuracy'] > max_accu or max_accu == -1:
        #     print('accuracy = ', accuracy, '>= min_accuracy(', max_accu, '), saving model...')
        #     torch.save([G_model, D_model], model_path)
        #     max_accu = accuracy
        max_res = GetMaxRes(max_res, res)
        print('Epoch ', epoch, 'loss = ', sum(Glosses) / len(Glosses), 'Valid = ', res , 'max accuracy=',
              max_accu)
        print('max res: ', max_res)
    # self.test()
    # showPlot(plot_losses)


def test(textData, model, datasetname, max_accuracy, eps = 1e-6):
    args['training'] = False
    right = 0
    total = 0
    total_with_rationale = 0
    dset = []

    pppt = False
    TP_c = np.zeros(3)
    FP_c = np.zeros(3)
    FN_c = np.zeros(3)
    TN_c = np.zeros(3)

    total_y_true = []
    total_y_score = []

    record_rationale_file = open(args['rootDir']+'hate_rationale.txt', 'w')
    with torch.no_grad():
        IOU_F1_gold_num = 0
        IOU_F1_pred_num =0
        IOU_F1_correct_num =0

        token_f1_sum = 0

        AUPRCs = []
        comprehensiveness = 0
        suff = 0
        for batch in textData.getBatches(datasetname):
            x = {}
            x['enc_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs))
            x['enc_len'] = batch.encoder_lens
            x['labels'] = autograd.Variable(torch.LongTensor(batch.label)).to(args['device'])
            # if args['model_arch'] in ['lstmibgan', 'lstmibgan_law']:
            #     x['labels'] = x['labels'][:, 0]

            output_probs, output_labels = model.predict(x)
            output_labels, sampled_words, wordsamplerate, prob_rational, output_all, output_norational = output_labels
            if not pppt:
                pppt = True
                pind = np.random.choice(x['enc_input'].size()[0])
                for w, choice in zip(batch.encoderSeqs[pind], sampled_words[pind]):
                    if choice[1] == 1:
                        print('<', textData.index2word[w], '>', end=' ')
                    else:
                        print(textData.index2word[w], end=' ')

                print('sample rate: ', wordsamplerate[0])
            print_rationales(record_rationale_file, textData, batch.encoderSeqs, sampled_words, batch.label, output_labels, batch.rationals)
            y = F.one_hot(torch.LongTensor(x['labels'].cpu().numpy()), num_classes=3)  # batch c
            y_onehot_T = y.to(args['device'])
            total_y_true.append(y)
            total_y_score.append(output_probs)
            y = y.bool().numpy()
            answer = output_labels.cpu().numpy()
            answer_onehot_T = F.one_hot(torch.LongTensor(answer), num_classes=3).to(args['device'])  # batch c
            answer = answer_onehot_T.cpu().bool().numpy()


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

            # rationales

            IOU_F1_pred_num += (sampled_words.sum(1) > 0).sum()
            tokenbatch_pred = sampled_words[:,:,1].sum(1, keepdim =True)

            # batch.rationals b 3 len
            IOU_F1_gold_num += (batch.rationals.sum(2).sum(1) > 0).sum()
            pred_rationales = sampled_words[:,:,1].unsqueeze(1).to(args['device']) # b 1 len
            intersect = (pred_rationales * batch.rationals).sum(2) # b 3
            joint = ((pred_rationales + batch.rationals) >0).sum(2)
            IOU = (intersect.float() /(joint.float()+1e-6)) > 0.5
            IOU = IOU.sum(1)
            IOU_F1_correct_num += (IOU > 0).sum()


            tokenbatch_gold = batch.rationals.sum(2) # batch 3
            tokenbatch_p = intersect / (tokenbatch_pred+1e-6)
            tokenbatch_r = intersect / (tokenbatch_gold+1e-6)
            tokenbatch_f1 = 2*tokenbatch_p*tokenbatch_r/ (tokenbatch_p + tokenbatch_r +1e-6)
            tokenbatch_f1,_ = torch.max(tokenbatch_f1, dim = 1) # batch
            total_with_rationale += (x['labels'] != 2).sum()
            token_f1_sum += tokenbatch_f1.sum()

            mask = torch.sign(x['enc_input']).to(args['device'])
            for gold_rationale, p_ret, gold_label, one_mask in zip(batch.rationals, prob_rational, batch.label, mask):  # 3 len    // len
                if gold_label != 2: # not normal:
                    largest_auc = 0
                    # numlen = one_mask.sum()
                    for i in range(3):
                        p_axis, r_axis, thres = precision_recall_curve(gold_rationale[i,:].cpu(), p_ret[:].cpu())
                        tmp_auc = auc(r_axis, p_axis)
                        if np.isnan(tmp_auc):
                            continue
                        largest_auc = tmp_auc if tmp_auc > largest_auc else largest_auc
                    AUPRCs.append(largest_auc)

            comprehensiveness += ((output_all - output_norational) * answer_onehot_T.float()).sum()
            suff += ((output_all - output_probs ) * y_onehot_T.float()).sum()

            for ind, c in enumerate(batch_correct):
                if not c:
                    dset.append((batch.encoderSeqs[ind], x['labels'][ind], output_labels[ind]))


    res ={}

    res['accuracy'] = right / total
    P_c = TP_c / (TP_c + FP_c)
    R_c = TP_c / (TP_c + FN_c)
    F_c = 2 * P_c * R_c / (P_c + R_c)
    res['F_macro'] = np.nanmean(F_c)
    res['MP'] = np.nanmean(P_c)
    res['MR'] = np.nanmean(R_c)

    total_y_true = torch.cat(total_y_true,dim = 0)
    total_y_score = torch.cat(total_y_score, dim = 0  )
    res['auroc'] = roc_auc_score(total_y_true.cpu(), total_y_score.cpu())


    res['ioup'] = (IOU_F1_correct_num / (IOU_F1_pred_num + 1e-6)).cpu().numpy()
    res['iour'] = (IOU_F1_correct_num / (IOU_F1_gold_num + 1e-6)).cpu().numpy()
    res['iouf1'] = 2 * res['ioup'] * res['iour'] / (res['ioup'] + res['iour'] + 1e-6)
    res['tokenF1'] = token_f1_sum / total_with_rationale
    res['AUPRC'] = torch.Tensor(AUPRCs).mean()

    res['comp'] = comprehensiveness / total
    res['suff'] = suff / total

    # P_c = TP_c / (TP_c + FP_c )
    # R_c = TP_c / (TP_c + FN_c )
    # F_c = 2 * P_c * R_c / (P_c + R_c )
    # res['F_macro_n ']= np.nanmean(F_c)
    # res['MP_n'] = np.nanmean(P_c)
    # res['MR_n'] = np.nanmean(R_c)

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

    args['training'] = True
    return res

def GetMaxRes(max_res, res):
    for key, value in res.items():

        if key in max_res:
            if key == 'suff':
                max_res[key]= min(max_res[key], res[key])
            else:
                max_res[key]= max(max_res[key], res[key])
        else:
            max_res[key]= res[key]
    return max_res


def print_rationales(record_file_handle, textData, encoderSeqs, sampled_words, label, output_labels, gold_rationales):
    for text, rationale_mask, gold_label, pred_label, gold_rat in zip(encoderSeqs, sampled_words,label, output_labels, gold_rationales):
        if len(gold_rat) > 0:
            rat_str = ''
            rat_str += (str(gold_label)+'\t' + str(pred_label) + '\t')
            if gold_label == pred_label:
                rat_str += ('Right\t')
            else:
                rat_str += ('Wrong\t')
            length = (torch.LongTensor(text) > 0).sum()
            samplerate = rationale_mask[:,1].sum() / length

            right = 0
            false_pos = 0
            true_neg = 0
            for w, choice, rat in zip(text, rationale_mask, gold_rat[0]):
                if choice[1] == 1:
                    if rat == 1:
                        rat_str += ('<$' + textData.index2word[w]+ '$> ')
                        right += 1
                    else:
                        rat_str += ('<' + textData.index2word[w]+ '> ')
                        false_pos += 1
                else:
                    if rat == 1:
                        rat_str += ('$' + textData.index2word[w]+'$ ')
                        true_neg += 1
                    else:
                        rat_str += (textData.index2word[w]+' ')

            # record_file_handle.write(str(right / (false_pos + right) )+ '\t')
            # record_file_handle.write(str(right /( true_neg + right)) + '\t')
            p=right / (false_pos + right+1e-6)
            r=right /( true_neg + right+1e-6)
            rat_str = str(samplerate)+ '\t'+ str(p)+ '\t' + str(r)+ '\t' + rat_str
            if p > 0.5 and r > 0.5 and length > 15:
                record_file_handle.write(rat_str + '\n')
