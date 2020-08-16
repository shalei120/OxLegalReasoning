import functools
print = functools.partial(print, flush=True)
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.init import _calculate_fan_in_and_fan_out
import numpy as np
import datetime, json
import time, math, gzip, random
from Encoder import Encoder
from rcnn_encoder import RCNNEncoder
from Decoder import Decoder
from Hyperparameters import args
from collections import namedtuple

from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, \
    ExponentialLR


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

class Vocabulary:
    """A vocabulary, assigns IDs to tokens"""

    def __init__(self, w2i,i2w):
        self.w2i = w2i
        self.i2w = i2w

class LSTM_IB_GAN_Model(nn.Module):
    """
    Implementation of a seq2seq model.
    Architecture:
        Encoder/decoder
        2 LTSM layers
    """

    def __init__(self, w2i, i2w, LM, i2v = None):
        """
        Args:
            args: parameters of the model
            textData: the dataset object
        """
        super(LSTM_IB_GAN_Model, self).__init__()
        print("Model creation...")
        self.vocab = Vocabulary(w2i,i2w)
        self.LM = LM
        self.word2index = w2i
        self.index2word = i2w
        self.max_length = args['maxLengthDeco']

        self.NLLloss = torch.nn.NLLLoss(reduction='none')
        self.CEloss = torch.nn.CrossEntropyLoss(reduction='none')

        # self.embedding = LM.embedding
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(i2v))
        self.embedding.weight.requires_grad = False

        self.arch = 'rcnn'
        if self.arch == 'lstm':
            self.encoder_all = Encoder(w2i, i2w, self.embedding).to(args['device'])
            self.encoder_select = Encoder(w2i, i2w, self.embedding, bidirectional = True).to(args['device'])
            self.encoder_mask = Encoder(w2i, i2w, self.embedding).to(args['device'])
        elif self.arch == 'rcnn':
            self.encoder_all = RCNNEncoder(args['embeddingSize'], args['hiddenSize']).to(args['device'])
            self.encoder_select = RCNNEncoder(args['embeddingSize'], args['hiddenSize'], bidirectional = True).to(args['device'])
            self.encoder_mask = RCNNEncoder(args['embeddingSize'], args['hiddenSize']).to(args['device'])

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        self.x_2_prob_z = nn.Sequential(
            nn.Linear(args['hiddenSize'] * 2, 2)
        ).to(args['device'])
        self.z_to_fea_best = nn.Linear(args['hiddenSize'], args['hiddenSize']).to(args['device'])
        self.z_to_fea_sample = nn.Linear(args['hiddenSize'], args['hiddenSize']).to(args['device'])

        self.review_scorer_best = nn.Sequential(
            nn.Linear(args['hiddenSize'], 1),
            nn.Sigmoid()
        ).to(args['device'])

        self.review_scorer_sample = nn.Sequential(
            nn.Linear(args['hiddenSize'],1),
            nn.Sigmoid()
        ).to(args['device'])
        # self.dropout = nn.Dropout(0.05)

        self.criterion = nn.MSELoss(reduction='none')


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

    @property
    def z(self):
        return self.sampled_seq

    def build(self, x, eps=1e-6):
        '''
        :param encoderInputs: [batch, enc_len]
        :param decoderInputs: [batch, dec_len]
        :param decoderTargets: [batch, dec_len]
        :return:
        '''

        # print(x['enc_input'])
        self.encoderInputs = x['enc_input']
        self.encoder_lengths = x['enc_len'] #torch.LongTensor(x['enc_len']).to(args['device'])
        self.scores = x['labels']
        self.batch_size = self.encoderInputs.size()[0]
        self.seqlen = self.encoderInputs.size()[1]
        optional = {}
        # best
        mask = torch.sign(self.encoderInputs).float()
        if self.arch ==  'rcnn':
            self.encoderInputs = self.embedding(self.encoderInputs)
            en_outputs, en_final = self.encoder_all(self.encoderInputs, mask, self.encoder_lengths)  # batch seq hid
        elif self.arch ==  'lstm':
            en_outputs, en_state = self.encoder_all(self.encoderInputs, self.encoder_lengths)  # batch seq hid

        z_nero_best = self.z_to_fea_best(en_outputs)
        z_nero_best, _ = torch.max(z_nero_best, dim=1)  # batch hid

        # z_nero_best = self.dropout(z_nero_best)
        output_all = self.review_scorer_best(z_nero_best)  # batch 5
        # recon_loss_all = (output_all - self.scores) ** 2
        # recon_loss_mean_all = recon_loss_all[:, args['aspect']]
        loss_mat_best = self.criterion(output_all, self.scores[:, args['aspect']].unsqueeze(1))
        loss_vec_best = loss_mat_best.mean(1)


        # sample
        if self.arch ==  'lstm':
            en_outputs_select, en_state = self.encoder_select(self.encoderInputs, self.encoder_lengths)  # batch seq hid
        elif self.arch ==  'rcnn':
            en_outputs_select, en_final = self.encoder_select(self.encoderInputs, mask, self.encoder_lengths)  # batch seq hid

        # print(en_outputs.size())
        z_logit = self.x_2_prob_z(en_outputs_select)  # batch seq 2
        self.z_prob = z_prob = self.softmax(z_logit) # batch seq 2

        if self.training:
            z_logit_fla = z_logit.reshape((self.batch_size * self.seqlen, 2))
            sampled_seq, sampled_seq_soft = self.gumbel_softmax(z_logit_fla) # batch seq  //0-1
            sampled_seq = sampled_seq.reshape((self.batch_size, self.seqlen, 2))
            sampled_seq_soft = sampled_seq_soft.reshape((self.batch_size, self.seqlen, 2))
            sampled_seq = sampled_seq * mask.unsqueeze(2)
            sampled_seq_soft = sampled_seq_soft * mask.unsqueeze(2)
            # sampled_seq = sampled_seq.detach()
        else:
            print('Evaluating...')
            sampled_seq = (z_prob >= 0.5).float()
        # print(sampled_seq)

        # sampled_word = self.encoderInputs * (sampled_seq[:,:,1])  # batch seq

        if self.arch ==  'lstm':
            en_outputs_masked, en_state = self.encoder_mask(self.encoderInputs, self.encoder_lengths,
                                                        sampled_seq[:, :, 1])  # batch seq hid
            z_nero_sampled = self.z_to_fea(en_outputs_masked[:,-1,:])
        elif self.arch ==  'rcnn':
            emb = self.encoderInputs * sampled_seq[:, :, 1].unsqueeze(2)
            en_outputs_masked, en_final = self.encoder_mask(emb, sampled_seq[:, :, 1], self.encoder_lengths)  # batch seq hid

            z_nero_sampled = en_final
        # z_nero_sampled, _ = torch.max(s_w_feature, dim=1)  # batch hid

        # I_x_z = torch.mean(-torch.log(z_prob[:, :, 0] + eps), dim = 1) * 100
        # I_x_z = sampled_seq[:, :, 1].sum(1)
        I_x_z = (z_prob * torch.log(z_prob / torch.FloatTensor([0.9999,0.0001]).unsqueeze(0).unsqueeze(1).to(args['device']))+eps).sum(2).sum(1) * 0.01
        # print(sampled_seq[0,:,:], torch.log(z_prob+eps)[0,:,:])

        self.sampled_seq = sampled_seq[:, :, 1]
        # logpz =  torch.sum(sampled_seq * torch.log(torch.min(z_prob+eps, torch.FloatTensor([1.0]).to(args['device']))), dim = 2)
        # logpz = (logpz*mask).sum(dim = 1)

        logp_z0 = torch.log(z_prob[:, :, 0])  # [B,T], log P(z = 0 | x)
        logp_z1 = torch.log(z_prob[:, :, 1])  # [B,T], log P(z = 1 | x)
        logpz = torch.where(sampled_seq[:, :, 1] == 0, logp_z0, logp_z1)
        logpz = mask * logpz
        # print(logpz)
        # print(I_x_z)
        # en_hidden, en_cell = en_state   #2 batch hid
        # omega = torch.sum(torch.abs(sampled_seq[:,:-1,1] - sampled_seq[:,1:,1]), dim = 1)
        omega = self.LM.LMloss(sampled_seq[:, :, 1], x['enc_input'])
        # omega = torch.mean(omega)
        # z_nero_sampled = self.dropout(z_nero_sampled)
        output = self.review_scorer_sample(z_nero_sampled)  # batch aspectnum
        loss_mat = self.criterion(output, self.scores[:,args['aspect']].unsqueeze(1))
        # recon_loss = (output - self.scores)**2
        # recon_loss_mean = recon_loss[:, args['aspect']]
        loss_vec = loss_mat.mean(1)  # [B]
        optional["mse_sp"] = loss_vec.mean().item()  # [1]
        optional['mse_best'] = loss_vec_best.mean().item()
        optional['I_x_z'] = I_x_z.mean().item()
        optional['zdiff'] = omega.mean().item()
        try:
            num_0, num_c, num_1, total = self.get_z_stats(self.sampled_seq, mask)
            optional["p0"] = num_0 / float(total)
            optional["p1"] = num_1 / float(total)
            optional["selected"] = optional["p1"]
        except:
            print(optional)
            exit(0)

        return loss_vec,  0.0003 * I_x_z, 0.06*omega , loss_vec_best , z_nero_best, z_nero_sampled, output, self.sampled_seq, logpz, optional
        # return  0, 0,0, 0, 0, 0,self.sampled_seq, 0,0, 0 , main_loss, optional

    def get_z_stats(self, z=None, mask=None, eps = 1e-6):
        """
        Computes statistics about how many zs are
        exactly 0, continuous (between 0 and 1), or exactly 1.
        :param z:
        :param mask: mask in [B, T]
        :return:
        """

        z = torch.where(mask>0, z, z.new_full([1], 1e2))

        num_0 = (z < eps).sum().item()
        num_c = (( eps <z) & (z < 1. -eps)).sum().item()
        num_1 = ((z > 1.-eps) & (z < 1 + eps)).sum().item()

        total = num_0 + num_c + num_1
        mask_total = mask.sum().item()
        try:
            assert total == mask_total, "total mismatch"
        except:
            print(z, mask)
            print(num_0,num_1,num_c, total, mask_total)
            assert total == mask_total, "total mismatch"
        return num_0, num_c, num_1, mask_total

    def forward(self, x):
        losses,I, om,best_loss, z_nero_best, z_nero_sampled, _, _,pz, optional = self.build(x)
        return losses,I, om,best_loss, z_nero_best, z_nero_sampled,pz, optional

    def predict(self, x):
        _, _,_,_, _, _, output,sampled_words, _, _  = self.build(x)
        return output, sampled_words

def xavier_uniform_n_(w, gain=1., n=4):
    """
    Xavier initializer for parameters that combine multiple matrices in one
    parameter for efficiency. This is e.g. used for GRU and LSTM parameters,
    where e.g. all gates are computed at the same time by 1 big matrix.
    :param w:
    :param gain:
    :param n:
    :return:
    """
    with torch.no_grad():
        fan_in, fan_out = _calculate_fan_in_and_fan_out(w)
        assert fan_out % n == 0, "fan_out should be divisible by n"
        fan_out = fan_out // n
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        nn.init.uniform_(w, -a, a)

def initialize_model_(model):
    """
    Model initialization.
    :param model:
    :return:
    """
    print("Glorot init")
    for name, p in model.named_parameters():
        if name.startswith("embed") or "lagrange" in name:
            print("{:10s} {:20s} {}".format("unchanged", name, p.shape))
        elif "lstm" in name and len(p.shape) > 1:
            print("{:10s} {:20s} {}".format("xavier_n", name, p.shape))
            xavier_uniform_n_(p)
        elif len(p.shape) > 1:
            print("{:10s} {:20s} {}".format("xavier", name, p.shape))
            torch.nn.init.xavier_uniform_(p)
        elif "bias" in name:
            print("{:10s} {:20s} {}".format("zeros", name, p.shape))
            torch.nn.init.constant_(p, 0.)
        else:
            print("{:10s} {:20s} {}".format("unchanged", name, p.shape))


def train(textData, LM, i2v=None, model_path=args['rootDir'] + '/chargemodel_LSTM_IB_GAN.mdl', print_every=50, plot_every=10,
          learning_rate=0.001, n_critic=5, eps = 1e-6):
    test_data = beer_annotations_reader('../beer/annotations.json', aspect=0)
    start = time.time()
    plot_losses = []
    print_Gloss_total = 0  # Reset every print_every
    plot_Gloss_total = 0  # Reset every plot_every
    print_Dloss_total = 0  # Reset every print_every
    plot_Dloss_total = 0  # Reset every plot_every
    G_model = LSTM_IB_GAN_Model(textData.word2index, textData.index2word, LM, i2v).to(args['device'])
    D_model = Discriminator().to(args['device'])

    print(type(textData.word2index))

    G_optimizer = optim.Adam(G_model.parameters(), lr=0.0004, weight_decay=2e-6)
    D_optimizer = optim.Adam(D_model.parameters(), lr=0.0004, weight_decay=2e-6)
    initialize_model_(G_model)
    initialize_model_(D_model)
    if args["scheduler"] == "plateau":
        scheduler = ReduceLROnPlateau(
            G_optimizer, mode='min', factor=args["lr_decay"],
            patience=args["patience"],
            threshold=args["threshold"], threshold_mode='rel',
            cooldown=args["cooldown"], verbose=True, min_lr=args["min_lr"])
    elif args["scheduler"] == "exponential":
        scheduler = ExponentialLR(G_optimizer, gamma=args["lr_decay"])
    elif args["scheduler"] == "multistep":
        milestones = args["milestones"]
        print("milestones (epoch):", milestones)
        scheduler = MultiStepLR(
            G_optimizer, milestones=milestones, gamma=args["lr_decay"])
        scheduler_D = MultiStepLR(
            D_optimizer, milestones=milestones, gamma=args["lr_decay"])
    else:
        raise ValueError("Unknown scheduler")

    iter = 1
    batches = textData.getBatches()
    n_iters = len(batches)
    print('niters ', n_iters)

    args['trainseq2seq'] = False

    max_p = -1
    # accuracy = test(textData, G_model, 'test', max_p)

    test2(G_model, test_data)
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

            x = {}
            x['enc_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs)).to(args['device'])
            x['enc_len'] = torch.LongTensor(batch.encoder_lens).to(args['device'])
            x['labels'] = autograd.Variable(torch.FloatTensor(batch.label)).to(args['device'])

            G_model.train()
            G_model.zero_grad()
            D_model.zero_grad()

            Gloss_pure, I, om, Gloss_best, z_nero_best, z_nero_sampled,logpz, optional = G_model(x)  # batch seq_len outsize
            Dloss = -torch.mean(torch.log(D_model(z_nero_best))) + torch.mean(torch.log(D_model(z_nero_sampled.detach())))
            # Dloss = torch.Tensor([0])
            Dloss.backward(retain_graph=True)
            #
            torch.nn.utils.clip_grad_norm_(D_model.parameters(), args['clip'])
            #
            D_optimizer.step()

            # if i % n_critic == 0:
            # -----------------
            #  Train Generator
            # -----------------

            G_model.zero_grad()
            D_model.zero_grad()
            # print(Gloss_best.size(), Gloss_pure.size(), D_model(z_nero_sampled).size(), logpz.size(), regu.size())
            # print(torch.log(D_model(z_nero_sampled)+eps), logpz)
            #- torch.log(D_model(z_nero_sampled)+eps) Gloss_best.mean() +
            G_ganloss = torch.log(D_model(z_nero_sampled).squeeze())
            Gloss = Gloss_best.mean() + Gloss_pure.mean() +10*I.mean()+om.mean()\
                    + ((Gloss_pure.detach() +I.detach()+om.detach()) * logpz.sum(1)).mean() -G_ganloss.mean()
            # cost_vec = Gloss_pure.detach() + regu
            # Gloss = Gloss_pure.mean() + (cost_vec * logpz.sum(1)).mean(0)


            # print(Gloss_best.mean() , Gloss_pure.mean(),((Gloss_pure.detach() +regu ) * logpz).mean())
            # Gloss = torch.mean(Gloss_best + Gloss_pure - torch.log(D_model(z_nero_sampled)+eps))
            Gloss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(G_model.parameters(), args['clip'])
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

        MSEloss, total_prec, samplerate = test(textData, G_model, 'test', max_p)
        cur_lr = scheduler.optimizer.param_groups[0]["lr"]
        if cur_lr > args["min_lr"]:
            if isinstance(scheduler, MultiStepLR):
                scheduler.step()
            elif isinstance(scheduler, ExponentialLR):
                scheduler.step()

        cur_lr = scheduler.optimizer.param_groups[0]["lr"]
        print("#lr", cur_lr)
        scheduler.optimizer.param_groups[0]["lr"] = max(args["min_lr"],
                                                        cur_lr)

        test2(G_model, test_data)
        if total_prec > max_p or max_p == -1:
            print('total_prec = ', total_prec, '>= max_p(', max_p, '), saving model...')
            torch.save([G_model, D_model], model_path)
            max_p = total_prec

        print('Epoch ', epoch, 'loss = ', sum(Glosses) / len(Glosses), sum(Dlosses) / len(Dlosses), 'Valid  mseloss= ', MSEloss, 'prec=',total_prec, 'sample=',samplerate, 'max prec=',
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
    model.eval()
    with torch.no_grad():
        for batch in textData.getBatches(datasetname):
            x = {}
            x['enc_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs)).to(args['device'])
            x['enc_len'] = torch.LongTensor(batch.encoder_lens).to(args['device'])
            x['labels'] = autograd.Variable(torch.FloatTensor(batch.label)).to(args['device'])

            output_probs, sampled_words = model.predict(x)
            if not pppt:
                pppt = True
                for w, choice in zip(batch.encoderSeqs[0], sampled_words):
                    if choice[1] == 1:
                        print('<',textData.index2word[w],'>', end=' ')
                    else:
                        print(textData.index2word[w], end=' ')


            batch_correct = (output_probs.cpu().numpy() - x['labels'].cpu().numpy())**2
            batch_correct = batch_correct[:, args['aspect']]
            # print(output_labels.size(), torch.LongTensor(batch.label).size())
            # right += sum(batch_correct[:, args['aspect']])
            MSEloss = (MSEloss * total + batch_correct.sum(axis = 0) ) / (total + x['enc_input'].size()[0])

            seqlen = torch.sign(x['enc_input']).float().sum(dim = 1) # batch
            prec = 0.0
            for i, b in enumerate(batch.encoder_lens):
                right = 0
                for w, choice in zip(batch.encoderSeqs[i], sampled_words[i]):
                    if choice == 1 and w in batch.rationals[i][args['aspect']]:
                        right += 1
                prec += right / seqlen[i]
            total_prec =( total_prec * total + prec) / (total + x['enc_input'].size()[0])
            # samplerate = (samplerate * total + wordsamplerate.sum(dim = 0))/ (total + x['enc_input'].size()[0])

            total += x['enc_input'].size()[0]


            # for ind, c in enumerate(batch_correct):
            #     if not c:
            #         dset.append((batch.encoderSeqs[ind], x['labels'][ind], output_labels[ind]))


    return MSEloss, total_prec, 0

def beer_reader(path, aspect=-1, max_len=0):
    """
    Reads in Beer multi-aspect sentiment data
    :param path:
    :param aspect: which aspect to train/evaluate (-1 for all)
    :return:
    """

    BeerExample = namedtuple("Example", ["tokens", "scores"])
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            scores = list(map(float, parts[:5]))

            if aspect > -1:
                scores = [scores[aspect]]

            tokens = parts[5:]
            if max_len > 0:
                tokens = tokens[:max_len]
            yield BeerExample(tokens=tokens, scores=scores)


def beer_annotations_reader(path, aspect=-1):
    """
    Reads in Beer annotations from json
    :param path:
    :param aspect: which aspect to evaluate
    :return:
    """
    BeerTestExample = namedtuple("Example", ["tokens", "scores", "annotations"])
    examples = []
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            tokens = data["x"]
            scores = data["y"]
            annotations = [data["0"], data["1"], data["2"],
                           data["3"], data["4"]]

            if aspect > -1:
                scores = [scores[aspect]]
                annotations = [annotations[aspect]]

            ex = BeerTestExample(
                tokens=tokens, scores=scores, annotations=annotations)
            examples.append(ex)
    return examples

def get_minibatch(data, batch_size=256, shuffle=False):
    """Return minibatches, optional shuffling"""

    if shuffle:
        print("Shuffling training data")
        random.shuffle(data)  # shuffle training data each epoch

    batch = []

    # yield minibatches
    for example in data:
        batch.append(example)

        if len(batch) == batch_size:
            yield batch
            batch = []

    # in case there is something left
    if len(batch) > 0:
        yield batch

def pad(tokens, length, pad_value=0):
    """add padding 1s to a sequence to that it has the desired length"""
    return tokens + [pad_value] * (length - len(tokens))

def prepare_minibatch(mb, vocab, device=None, sort=True):
    """
    Minibatch is a list of examples.
    This function converts words to IDs and returns
    torch tensors to be used as input/targets.
    """
    # batch_size = len(mb)
    lengths = np.array([len(ex.tokens) for ex in mb])
    maxlen = lengths.max()
    reverse_map = None

    # vocab returns 0 if the word is not there
    x = [pad([vocab.w2i.get(t, 3) for t in ex.tokens], maxlen) for ex in mb]
    y = [ex.scores for ex in mb]

    x = np.array(x)
    y = np.array(y, dtype=np.float32)

    if sort:  # required for LSTM
        sort_idx = np.argsort(lengths)[::-1]
        x = x[sort_idx]
        y = y[sort_idx]

        # create reverse map
        reverse_map = np.zeros(len(lengths), dtype=np.int32)
        for i, j in enumerate(sort_idx):
            reverse_map[j] = i

    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)

    return x, y, reverse_map

def decorate_token(t, z_):
    dec = "**" if z_ == 1 else "__" if z_ > 0 else ""
    return dec + t + dec

def evaluate_rationale(model, data, aspect=None, batch_size=256, device=None,
                       path=None):
    """Precision on annotated rationales.
    This works in a simple way:
    We have a predicted vector z
    We have a gold annotation  z_gold
    We take a logical and (to intersect the two)
    We sum the number of words in the intersection and divide by the total
    number of selected words.
    """

    assert aspect is not None, "provide aspect"
    assert device is not None, "provide device"

    # if not hasattr(model, "z"):
    #     print('No z')
    #     return

    if path is not None:
        ft = open(path, mode="w", encoding="utf-8")
        fz = open(path + ".z", mode="w", encoding="utf-8")

    model.eval()  # disable dropout
    sent_id, correct, total, macro_prec_total, macro_n = 0, 0, 0, 0, 0
    macro_rec_total , total_should_match, macro_F_total = 0,0, 0
    for mb in get_minibatch(data, batch_size=batch_size, shuffle=False):
        x, targets, reverse_map = prepare_minibatch(
            mb, model.vocab, device=device, sort=True)
        with torch.no_grad():
            xmap = {'enc_input': x,
                    'enc_len': torch.sign(x).long().sum(1).detach(),
                    'labels': targets}
            # logits = model(xmap)
            # print('mbing 1')
            output, z= model.predict(xmap )
            # z = z[:,:,1]
            # print('mbing 2')
            # attention alphas
            # if hasattr(model, "alphas"):
            #     alphas = model.alphas
            # else:
            #     alphas = None

            # rationale z
            # if hasattr(model, "z"):
            # z = model.z  # [B, T]

            bsz, max_time = z.size()
            # else:
            #     z = None

        # the inputs were sorted to enable packed_sequence for LSTM
        # we need to reverse sort them so that they correspond
        # to the original order

        # reverse sort
        # alphas = alphas[reverse_map] if alphas is not None else None
        z = z[reverse_map] if z is not None else None  # [B,T]

        # evaluate each sentence in this minibatch
        for mb_i, ex in enumerate(mb):
            tokens = ex.tokens
            annotations = ex.annotations

            # assuming here that annotations only has valid ranges for the
            # current aspect
            if aspect > -1:
                assert len(annotations) == 1, "expected only 1 aspect"

            # if alphas is not None:
            #     alpha = alphas[mb_i][:len(tokens)]
            #     alpha = alpha[None, :]

            # z is [batch_size, time]
            if z is not None:

                z_ex = z[mb_i, :len(tokens)]  # i for minibatch example
                z_ex_nonzero = (z_ex > 0).float()
                z_ex_nonzero_sum = z_ex_nonzero.sum().item()

                # list of decorated tokens for this single example, to print
                example = []
                for ti, zi in zip(tokens, z_ex):
                    example.append(decorate_token(ti, zi))

                # write this sentence
                ft.write(" ".join(example))
                ft.write("\n")
                fz.write(" ".join(["%.4f" % zi for zi in z_ex]))
                fz.write("\n")

                # skip if no gold rationale for this sentence
                if aspect >= 0 and len(annotations[0]) == 0:
                    continue

                # compute number of matching tokens & precision
                matched = sum(1 for i, zi in enumerate(z_ex) if zi > 0 and
                              any(interval[0] <= i < interval[1]
                                  for a in annotations for interval in a))

                should_matched = sum([interval[1]-interval[0] for a in annotations for interval in a])

                precision = matched / (z_ex_nonzero_sum + 1e-9)
                recall = matched /(should_matched+ 1e-9)
                F = 2*precision*recall / (precision + recall + 1e-9)

                macro_prec_total += precision
                macro_rec_total += recall
                macro_F_total += F

                correct += matched
                total += z_ex_nonzero_sum
                total_should_match += should_matched
                if z_ex_nonzero_sum > 0:
                    macro_n += 1

                # print(matched, end="\t")

            sent_id += 1
    # print()
    # print("new correct", correct, "total", total)

    precision = correct / (total + 1e-9)
    recall = correct / (total_should_match + 1e-9)
    macro_precision = macro_prec_total / (float(macro_n) + 1e-9)
    macro_recall = macro_rec_total / (float(macro_n) + 1e-9)
    macro_F = macro_F_total / (float(macro_n) + 1e-9)

    try:
        ft.close()
        fz.close()
    except IOError:
        print("Error closing file(s)")

    return precision, macro_precision, recall,macro_recall, macro_F

def test2(model,test_data):
    test_precision, test_macro_prec, test_recall, test_macro_recall,macro_F = evaluate_rationale(model, test_data, aspect=0,device=args['device'], path='./record.txt', batch_size=256)

    print('Rational: P_micro: ', test_precision, 'P_macro: ', test_macro_prec)
    print('Rational: R_micro: ', test_recall, 'R_macro: ', test_macro_recall)
    print('Rational: F_micro: ', 2*test_precision*test_recall, 'F_macro: ', macro_F)
