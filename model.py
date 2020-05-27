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

class Model(nn.Module):
    """
    Implementation of a seq2seq model.
    Architecture:
        Encoder/decoder
        2 LTSM layers
    """

    def __init__(self, args,w2i, i2w):
        """
        Args:
            args: parameters of the model
            textData: the dataset object
        """
        super(Model, self).__init__()
        print("Model creation...")

        self.word2index = w2i
        self.index2word = i2w
        self.max_length = args['maxLengthDeco']

        self.NLLloss = torch.nn.NLLLoss(reduction = 'none')
        self.CEloss =  torch.nn.CrossEntropyLoss(reduction = 'none')

        self.embedding = nn.Embedding(args['vocabularySize'], args['embeddingSize']).to(self.device)

        self.encoder = Encoder(args,w2i, i2w, self.embedding, self.device).to(self.device)
        self.decoder = Decoder(args,w2i, i2w, self.embedding, self.device, hidden_size =  args['hiddenSize']).to(self.device)

        self.decompose_hidden2style = nn.Linear(args['hiddenSize'], args['style_len'], bias = True).to(device)
        self.decompose_cell2style = nn.Linear(args['hiddenSize'], args['style_len'], bias = True).to(device)
        self.decompose_hidden2content = nn.Linear(args['hiddenSize'], args['content_len'], bias = True).to(device)
        self.decompose_cell2content = nn.Linear(args['hiddenSize'], args['content_len'], bias = True).to(device)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = -1)

        self.adapt_unit_style2h = nn.Linear( args['style_len'],
                                    args['hiddenSize'], bias = True).to(self.device)
        self.adapt_unit_style2c = nn.Linear( args['style_len'],
                                    args['hiddenSize'], bias = True).to(self.device)
        self.adapt_unit_content2h = nn.Linear(args['content_len'] ,
                                    args['hiddenSize'], bias = True).to(self.device)
        self.adapt_unit_content2c = nn.Linear(args['content_len'] ,
                                    args['hiddenSize'], bias = True).to(self.device)

        self.style1_mu = Parameter(torch.rand(args['style1_labelnum'], args['style_len']*2* args['numLayers'], device = self.device))
        self.style1_logvar = Parameter(torch.rand(args['style1_labelnum'], args['style_len']*2* args['numLayers'], device = self.device))
        self.style1_var = torch.exp(self.style1_logvar)

        self.style1_classifier = nn.Linear(args['style_len'] * 2 * args['numLayers'], args['style1_labelnum'], bias = True).to(self.device)


        self.labelmu = Parameter(torch.rand(args['style1_labelnum'],args['numLayers'], args['content_len']*2, device = self.device))
        self.labellogvar = Parameter(torch.rand(args['style1_labelnum'], args['numLayers'], args['content_len']*2, device = self.device))
        self.labelsigma = torch.exp(self.labellogvar)

        self.mainparams = list(self.decompose_hidden2style.parameters()) + list(self.decompose_cell2style.parameters()) \
                          + list(self.adapt_unit_style2h.parameters()) + list(self.adapt_unit_style2c.parameters()) \
                          + [self.style1_mu ,self.style1_logvar] + list(self.style1_classifier.parameters()) + [self.labelmu, self.labellogvar]

        self.sc_disentangle_params = [self.labelmu, self.labellogvar]

    def sample_z(self, mu, log_var,batch_size):
        eps = Variable(torch.randn(batch_size, args['style_len']*2* args['numLayers'])).to(self.device)
        return mu + torch.einsum('ba,ba->ba', torch.exp(log_var/2),eps)

    def forward(self, x, factor):
        '''
        :param encoderInputs: [batch, enc_len]
        :param decoderInputs: [batch, dec_len]
        :param decoderTargets: [batch, dec_len]
        :return:
        '''

        # print(x['enc_input'])
        self.encoderInputs = x['enc_input'].to(device)
        self.encoder_lengths = x['enc_len']
        self.decoderInputs = x['dec_input'].to(device)
        self.decoder_lengths = x['dec_len']
        self.decoderTargets = x['dec_target'].to(device)
        self.classifyLabels = x['labels'].to(device)
        self.batch_size = self.encoderInputs.size()[0]

        en_state = self.encoder(self.encoderInputs, self.encoder_lengths)
        style_emb, content_emb = self.decompose(en_state)
        self.style_emb = style_emb
        self.content_emb = content_emb

        # print('style_emb: ', style_emb.size())

        self.batch_mus = torch.index_select(self.style1_mu, 0, self.classifyLabels)
        self.batch_vars = torch.index_select(self.style1_var, 0, self.classifyLabels)


        if args['style-center'] and not args['trainseq2seq']:
            style_z = self.sample_z(self.batch_mus, torch.log(self.batch_vars), self.batch_size)
            style_z = style_z.view(self.batch_size, args['numLayers'], args['style_len']*2)
            style_z = torch.transpose(style_z, 0,1)

        if args['style-center'] and not args['trainseq2seq']:
            h,c = self.recombine(style_z, content_emb)
        else:
            h,c = self.recombine(style_emb, content_emb)

        de_input_state = en_state if args['trainseq2seq'] else (h,c)
        de_outputs = self.decoder(de_input_state, self.decoderInputs, self.decoder_lengths, self.decoderTargets)

        perplexity_train = self.Get_perplexity_train(self.decoderInputs, self.decoder_lengths, self.decoderTargets)

        recon_loss = self.CEloss(torch.transpose(de_outputs, 1, 2), self.decoderTargets)
        # loss = - torch.gather(predictions, 2, torch.unsqueeze(target_variable, 2))
        mask = torch.sign(self.decoderTargets.float())
        # print(loss.shape,mask.shape)
        recon_loss = torch.squeeze(recon_loss) * mask

        # recon_loss_mean = torch.mean(recon_loss, dim = -1) #* factor #* (100 if epoch <10 else 1)
        recon_loss_mean = torch.mean(recon_loss)*300  #* factor #* (100 if epoch <10 else 1)

        # scloss, prob_nll_loss = 0 ,0
        scloss, prob_nll_loss = self.SC_disentangle_loss(content_emb) #self.batch_dot_product_L2(style_emb, content_emb)# if not args['trainseq2seq'] else 0
        scloss *= 1000  # *5
        prob_nll_loss *= 150
        recon_loss_mean += scloss   # if not args['trainseq2seq'] else 0

        recon_loss_mean += prob_nll_loss
        pl = 0
        cl = 0
        # style_emb : numlayers * batch * len
        if args['style-center'] and not args['trainseq2seq']:
            batch_style1_mu = self.style1_mu.view(args['style1_labelnum'], args['numLayers'], 2*args['style_len'])
            batch_style1_mu = torch.unsqueeze(batch_style1_mu, 2)
            batch_style1_var = self.style1_var.view(args['style1_labelnum'], args['numLayers'], 2*args['style_len'])
            batch_style1_var = torch.unsqueeze(batch_style1_var, 2)

            style_emb_per_dist = torch.unsqueeze(style_emb, 0).repeat(args['style1_labelnum'], 1,1,1)
            prob_style1_styleemb = torch.exp(-(style_emb_per_dist - batch_style1_mu)**2/(2 * batch_style1_var+ 0.0000001)) / torch.sqrt(2 * np.pi * batch_style1_var)
                            # labelnum * numlayer * batch * len
            prob_style1_styleemb = torch.log(prob_style1_styleemb + 0.0000001)
            prob_style1_styleemb = torch.mean(prob_style1_styleemb, dim = 3)
            prob_style1_styleemb = torch.mean(prob_style1_styleemb, dim = 1) # labelnum * batch

            prob_style1_styleemb = torch.transpose(prob_style1_styleemb, 0,1)#  batch * labelnum
            # print(prob_style1_styleemb)
            prob_style1_styleemb = torch.exp(prob_style1_styleemb) # followed by logsoftmax
            # prob_style1_styleemb = torch.tanh(prob_style1_styleemb) # quasi-prob


            prob_style1_styleemb_mean = torch.mean(self.CEloss(prob_style1_styleemb, self.classifyLabels))
            # print(prob_style1_styleemb_mean)

            style_1_sample = self.sample_z(self.batch_mus, torch.log(self.batch_vars), self.batch_size)

            y = self.style1_classifier(style_1_sample)
            # y = self.style1_classifier(self.batch_mus)
            classify_loss = torch.mean(self.CEloss(y, self.classifyLabels))

            KL_loss1 = torch.sum(0.5 * (torch.exp(self.style1_logvar) + self.style1_mu ** 2  - 1 - self.style1_logvar))

            # #L_disc
            # classify_loss = 0

            loss_mean = recon_loss_mean + prob_style1_styleemb_mean*20 + classify_loss  + self.styletype_dissolve()*1  + 0.01 * KL_loss1
            # loss_mean = recon_loss_mean  + prob_style1_styleemb_mean + classify_loss  + self.styletype_dissolve() + 0.01 * KL_loss1
            pl = prob_style1_styleemb_mean
            cl = classify_loss
        else:
            style_emb = torch.transpose(style_emb, 0,1)
            style_emb = style_emb.contiguous().view(self.batch_size, -1)
            y = self.style1_classifier(style_emb)
            classify_loss = torch.mean(self.CEloss(y, self.classifyLabels))


            # #L_disc
            # classify_loss = 0

            loss_mean = recon_loss_mean + classify_loss #+ prob_style1_styleemb_mean

        loss_mean += perplexity_train * 300

        return loss_mean, pl, cl, scloss, prob_nll_loss, perplexity_train

    def styletype_dissolve(self):

        labelmu1 = self.style1_mu[0,:]
        labellogvar1 = self.style1_logvar[0,:]
        labelmu1_repeat = labelmu1.repeat(args['style1_labelnum'], 1)
        labellogvar1_repeat = labellogvar1.repeat(args['style1_labelnum'], 1)


        styletypedissolve_loss = 0.5*(torch.exp(labellogvar1_repeat - self.style1_logvar) + (labelmu1_repeat - self.style1_mu)**2/torch.exp(self.style1_logvar) - 1 + self.style1_logvar - labellogvar1_repeat)
        styletypedissolve_loss = torch.mean(styletypedissolve_loss, dim = 1)

        #maximize style type KL

        # return 100000 / torch.mean(styletypedissolve_loss)
        return  torch.mean(styletypedissolve_loss)


    def SC_disentangle_loss(self, content_emb):
        style_emb_per_dist = torch.unsqueeze(content_emb, 0).repeat(args['style1_labelnum'], 1,1,1)
        batch_c_mu = torch.unsqueeze(self.labelmu, 2)
        batch_c_sigma = torch.unsqueeze(self.labelsigma, 2)
        P_c_t = torch.exp(-(style_emb_per_dist - batch_c_mu) ** 2 / (2 * batch_c_sigma + 0.0000001)) / torch.sqrt(
            2 * np.pi * batch_c_sigma)# labelnum * numlayer * batch * len
        P_c_t = torch.log(P_c_t + 0.0000001)
        P_c_t = torch.mean(P_c_t, dim = 3)
        P_c_t = torch.mean(P_c_t, dim = 1) # labelnum * batch
        P_c_t = torch.transpose(P_c_t, 0,1).contiguous()#  batch * labelnum
        # P_c_t = torch.exp(P_c_t)
        # prob_style1_contentemb_mean = torch.mean(self.CEloss(P_c_t, self.classifyLabels))
        prob_style1_contentemb_mean = torch.mean(self.NLLloss(P_c_t, self.classifyLabels))

        # P_c_t_sm = self.softmax(P_c_t)#  batch * labelnum
        # entropy = - torch.sum(P_c_t_sm * torch.log(P_c_t_sm+ 0.0000001), dim = 1)
        # entropy = torch.mean(entropy)

        labelmu1 = self.labelmu[0,:,:]
        labellogvar1 = self.labellogvar[0,:,:]
        labelmu1_repeat = labelmu1.repeat(args['style1_labelnum'], 1,1)
        labellogvar1_repeat = labellogvar1.repeat(args['style1_labelnum'], 1,1)

        self.sc_loss = 0.5*(torch.exp(labellogvar1_repeat - self.labellogvar) + (labelmu1_repeat - self.labelmu)**2/torch.exp(self.labellogvar) - 1 + self.labellogvar - labellogvar1_repeat)
        # self.sc_loss = 0.5*((labelstd1_repeat / self.labelstd)**2 + (labelmu1_repeat - self.labelmu)**2/self.labelstd**2 - 1 + 2*torch.log(self.labelstd/labelstd1_repeat))
        self.sc_loss = torch.sum(self.sc_loss, dim = 2)
        self.sc_loss = torch.sum(self.sc_loss, dim = 1)
        # if np.isnan(self.sc_loss.data):
        #     sdf=0

        return torch.mean(self.sc_loss), prob_style1_contentemb_mean

    def predict(self, x, transfer = False):
        encoderInputs = x['enc_input']
        encoder_lengths = x['enc_len']
        classifyLabels = torch.LongTensor(x['labels']).to(device)

        batch_size = encoderInputs.size()[0]
        enc_len = encoderInputs.size()[1]

        en_state = self.encoder(encoderInputs, encoder_lengths)
        style_emb, content_emb = self.decompose(en_state)
        # print(style_emb.size())

        if transfer:
            batch_mus = torch.index_select(self.style1_mu, 0, 1-classifyLabels)
            batch_logvars = torch.index_select(self.style1_logvar, 0, 1-classifyLabels)
            style_emb = self.sample_z(batch_mus, batch_logvars, batch_size)
            style_emb = style_emb.view(batch_size, args['numLayers'], args['style_len']*2)
            style_emb = torch.transpose(style_emb, 0,1)

        h,c = self.recombine(style_emb, content_emb)

        de_input_state = en_state if args['trainseq2seq'] else (h,c)
        de_words = self.decoder.generate(de_input_state, en_state)

        style_emb = torch.transpose(style_emb, 0,1)
        style_emb = style_emb.contiguous().view(batch_size, -1)

        content_emb = torch.transpose(content_emb, 0,1)
        content_emb = content_emb.contiguous().view(batch_size, -1)

        label = self.style1_classifier(style_emb)
        label_c = self.style1_classifier(content_emb)

        return de_words, torch.argmax(label, dim = -1), torch.argmax(label_c, dim = -1)

    def Get_perplexity(self, x):
        encoderInputs = x['enc_input']
        encoder_lengths = x['enc_len']
        decoderInputs = x['dec_input'].to(device)
        decoder_lengths = x['dec_len']
        decoderTargets = x['dec_target'].to(device)
        classifyLabels = torch.LongTensor(x['labels']).to(device)

        batch_size = encoderInputs.size()[0]
        enc_len = encoderInputs.size()[1]

        # en_state = self.encoder(encoderInputs, encoder_lengths)
        # style_emb, content_emb = self.decompose(en_state)
        # h, c = self.recombine(style_emb, content_emb)
        #
        # de_input_state = en_state if args['trainseq2seq'] else (h, c)
        de_input_state= (torch.rand(args['dec_numlayer'], batch_size, args['hiddenSize'], device=self.device),
                           torch.rand(args['dec_numlayer'], batch_size, args['hiddenSize'], device=self.device))

        de_outputs = self.decoder(de_input_state, decoderInputs, decoder_lengths, decoderTargets)

        recon_loss = self.CEloss(torch.transpose(de_outputs, 1, 2), decoderTargets)

        mask = torch.sign(decoderTargets.float())
        # print(loss.shape,mask.shape)
        recon_loss = torch.squeeze(recon_loss) * mask

        recon_loss_mean = torch.mean(recon_loss, dim=-1)

        return list(np.asarray(torch.exp(recon_loss_mean).cpu()))

    def Get_perplexity_train(self, decoderInputs, decoder_lengths, decoderTargets):

        batch_size = decoderInputs.size()[0]

        de_input_state= (torch.rand(args['dec_numlayer'], batch_size, args['hiddenSize'], device=self.device),
                           torch.rand(args['dec_numlayer'], batch_size, args['hiddenSize'], device=self.device))

        de_outputs = self.decoder(de_input_state, decoderInputs, decoder_lengths, decoderTargets)

        recon_loss = self.CEloss(torch.transpose(de_outputs, 1, 2), decoderTargets)

        mask = torch.sign(decoderTargets.float())
        # print(loss.shape,mask.shape)
        recon_loss = torch.squeeze(recon_loss) * mask

        recon_loss_mean = torch.mean(recon_loss, dim=-1)

        return torch.mean(torch.exp(recon_loss_mean))

    def extract_useful(self, x):
        self.encoderInputs = x['enc_input']
        self.encoder_lengths = x['enc_len']

        self.batch_size = self.encoderInputs.size()[0]
        self.enc_len = self.encoderInputs.size()[1]

        en_state = self.encoder(self.encoderInputs, self.encoder_lengths)
        style_emb, content_emb = self.decompose(en_state)


        return style_emb, content_emb

    def sample_action(self, pdf):
        pdf = pdf.cpu().detach().data
        pdf = pdf.to(torch.float64)
        pdf = pdf / torch.sum(pdf, dim = 1, keepdim = True)
        batch_actions = []
        for b in range(pdf.size()[0]):
            rn = np.random.multinomial(1, pdf[b,:], size = 5)
            batch_actions.append(rn)
        batch_actions = torch.Tensor(batch_actions)  # batch 5 attsize
        batch_actions = torch.transpose(batch_actions, 0,1 )
        return batch_actions.to(torch.float32)



    def decompose(self, en_state):
        en_hidden, en_cell = en_state
        s = torch.cat([self.decompose_hidden2style(en_hidden), self.decompose_cell2style(en_cell)], dim = -1)
        c = torch.cat([self.decompose_hidden2content(en_hidden), self.decompose_cell2content(en_cell)], dim = -1)
        return self.tanh(s),self.tanh(c)


    def recombine(self, style_emb, content_emb):
        h_style, c_style = torch.split(style_emb, args['style_len'], dim = -1)
        h_content, c_content = torch.split(content_emb, args['content_len'], dim = -1)
        h1 = self.adapt_unit_style2h(h_style)
        h2 = self.adapt_unit_content2h(h_content)
        h = h1 + h2
        c1 = self.adapt_unit_style2c(c_style)
        c2 = self.adapt_unit_content2c(c_content)
        c = c1+c2
        # h1 = torch.cat([h_style, h_content], dim = -1)
        # c1= torch.cat([c_style, c_content], dim = -1)
        # h = self.adapt_unit_h(h1)
        # c = self.adapt_unit_c(c1)
        return self.tanh(h),self.tanh(c)

    def batch_dot_product_L2(self, style_emb, content_emb):
        bs = torch.transpose(style_emb, 0,1).contiguous()
        bs = bs.view(self.batch_size, -1)
        bc = torch.transpose(content_emb, 0,1).contiguous()
        bc = bc.view(self.batch_size, -1)
        dotps = torch.einsum('ba,ba->b', bs, bc)
        return torch.mean(dotps ** 2)




