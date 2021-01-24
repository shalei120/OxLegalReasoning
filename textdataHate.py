
import numpy as np
import nltk  # For tokenize
from tqdm import tqdm  # Progress bar
import pickle  # Saving the data
import math  # For float comparison
import os  # Checking file existance
import random, gzip
import string, copy
from nltk.tokenize import word_tokenize
import jieba
import json
from Hyperparameters import args
import torch
class Batch:
    """Struct containing batches info
    """
    def __init__(self):
        self.encoderSeqs = []
        self.encoder_lens = []
        self.label = []
        self.decoderSeqs = []
        self.targetSeqs = []
        self.decoder_lens = []
        self.rationals = []
        self.raw = []


class TextDataHate:
    """Dataset class
    Warning: No vocabulary limit
    """


    def __init__(self, corpusname, trainLM =False):

        """Load all conversations
        Args:
            args: parameters of the model
        """

        # Path variables
        if corpusname == 'cail':
            self.tokenizer = lambda x: list(jieba.cut(x))
        elif corpusname == 'beer' or corpusname == 'hate' :
            self.tokenizer = word_tokenize


        self.trainingSamples = []  # 2d array containing each question and his answer [[input,target]]
        self.datasets= self.loadCorpus()

        print('set')
        # Plot some stats:
        self._printStats(corpusname)

        if args['playDataset']:
            self.playDataset()

        self.batches = {}

    def _printStats(self, corpusname):
        print('Loaded {}: {} words, {} QA'.format(corpusname, len(self.word2index), len(self.trainingSamples)))


    def shuffle(self):
        """Shuffle the training samples
        """
        print('Shuffling the dataset...')
        random.shuffle(self.datasets['train'])

    def _createBatch(self, samples):
        """Create a single batch from the list of sample. The batch size is automatically defined by the number of
        samples given.
        The inputs should already be inverted. The target should already have <go> and <eos>
        Warning: This function should not make direct calls to args['batchSize'] !!!
        Args:
            samples (list<Obj>): a list of samples, each sample being on the form [input, target]
        Return:
            Batch: a batch object en
        """

        batch = Batch()
        batchSize = len(samples)

        # Create the batch tensor
        for i in range(batchSize):
            # Unpack the sample
            sen_ids, y, raw_sen, rational = samples[i]

            if len(sen_ids) > args['maxLengthEnco']:
                sen_ids = sen_ids[:args['maxLengthEnco']]

            batch.encoderSeqs.append(sen_ids)
            batch.encoder_lens.append(len(batch.encoderSeqs[i]))
            batch.label.append(y)

            batch.rationals.append(rational)
            batch.raw.append(raw_sen)
            # print(y)

        maxlen_enc = max(batch.encoder_lens)


        for i in range(batchSize):
            batch.encoderSeqs[i] = batch.encoderSeqs[i] + [self.word2index['[PAD]']] * (maxlen_enc - len(batch.encoderSeqs[i]))
            ration = torch.zeros([3, maxlen_enc])
            if len(batch.rationals[i]) >0:
                # try:
                ration_value = torch.LongTensor(batch.rationals[i])
                # except:
                #     sdv=0

                real_len = ration_value.size()[1]
                real_num = ration_value.size()[0]
                ration[:real_num,:real_len] = ration_value
            batch.rationals[i] = ration.to(args['device'])
        batch.rationals = torch.stack(batch.rationals) # batch 3 len

        return batch

    def getBatches(self, setname = 'train'):
        """Prepare the batches for the current epoch
        Return:
            list<Batch>: Get a list of the batches for the next epoch
        """
        if setname not in self.batches:
            # self.shuffle()

            batches = []
            print(setname, 'size:', len(self.datasets[setname]))
            def genNextSamples():
                """ Generator over the mini-batch training samples
                """
                for i in range(0, self.getSampleSize(setname), args['batchSize']):
                    yield self.datasets[setname][i:min(i + args['batchSize'], self.getSampleSize(setname))]

            # TODO: Should replace that by generator (better: by tf.queue)

            for index, samples in enumerate(genNextSamples()):
                # print([self.index2word[id] for id in samples[5][0]], samples[5][2])
                batch = self._createBatch(samples)
                batches.append(batch)

            self.batches[setname] = batches

        # print([self.index2word[id] for id in batches[2].encoderSeqs[5]], batches[2].raws[5])
        return self.batches[setname]

    def _createBatch_forLM(self, samples):
        """Create a single batch from the list of sample. The batch size is automatically defined by the number of
        samples given.
        The inputs should already be inverted. The target should already have <go> and <eos>
        Warning: This function should not make direct calls to args['batchSize'] !!!
        Args:
            samples (list<Obj>): a list of samples, each sample being on the form [input, target]
        Return:
            Batch: a batch object en
        """

        batch = Batch()
        batchSize = len(samples)

        # Create the batch tensor
        for i in range(batchSize):
            # Unpack the sample
            sen_ids = samples[i]
            if len(sen_ids) > args['maxLengthEnco']:
                sen_ids = sen_ids[:args['maxLengthEnco']]
            batch.decoderSeqs.append([self.word2index['[START_TOKEN]']] + sen_ids)
            batch.decoder_lens.append(len(batch.decoderSeqs[i]))
            batch.targetSeqs.append(sen_ids + [self.word2index['[END_TOKEN]']])

        # print(batch.decoderSeqs)
        # print(batch.decoder_lens)
        maxlen_dec = max(batch.decoder_lens)
        maxlen_dec = min(maxlen_dec, args['maxLengthEnco'])

        for i in range(batchSize):
            batch.decoderSeqs[i] = batch.decoderSeqs[i] + [self.word2index['[PAD]']] * (maxlen_dec - len(batch.decoderSeqs[i]))
            batch.targetSeqs[i] = batch.targetSeqs[i] + [self.word2index['[PAD]']] * (maxlen_dec - len(batch.targetSeqs[i]))

        return batch

    def paragraph2sentence(self, doclist):
        split_tokens = [self.word2index['.']]
        sen_list = []
        for sen_ids, y, raw_sen, rational in doclist:
            start = 0
            for ind, w in enumerate(sen_ids):
                if w in split_tokens:
                    sen_list.append(sen_ids[start:ind + 1])
                    start = ind + 1

            if start < len(sen_ids) - 1:
                sen_list.append(sen_ids[start:])

        return sen_list

    def getBatches_forLM(self, setname = 'train'):
        """Prepare the batches for the current epoch
        Return:
            list<Batch>: Get a list of the batches for the next epoch
        """
        if setname not in self.batches:
            self.shuffle()

            dataset_sen = self.paragraph2sentence(self.datasets[setname])
            sennum = len(dataset_sen)
            print(sennum)

            batches = []
            print(len(self.datasets[setname]))
            def genNextSamples():
                """ Generator over the mini-batch training samples
                """
                for i in range(0, sennum, args['batchSize']):
                    yield dataset_sen[i:min(i + args['batchSize'], sennum)]

            # TODO: Should replace that by generator (better: by tf.queue)

            for index, samples in enumerate(genNextSamples()):
                # print([self.index2word[id] for id in samples[5][0]], samples[5][2])
                batch = self._createBatch_forLM(samples)
                batches.append(batch)

            self.batches[setname] = batches

        # print([self.index2word[id] for id in batches[2].encoderSeqs[5]], batches[2].raws[5])
        return self.batches[setname]

    def getSampleSize(self, setname = 'train'):
        """Return the size of the dataset
        Return:
            int: Number of training samples
        """
        return len(self.datasets[setname])

    def getVocabularySize(self):
        """Return the number of words present in the dataset
        Return:
            int: Number of word on the loader corpus
        """
        return len(self.word2index)


    def loadCorpus(self, use_pretrained_vector = False):
        """Load/create the conversations data
        """
        self.basedir = '../HateXplain/Data/'
        self.corpus_file = self.basedir + 'dataset.json'
        self.corpus_file_split =  self.basedir + 'post_id_divisions.json'
        if use_pretrained_vector:
            self.vocfile = self.basedir + '/glove.840B.300d.txt'
            self.data_dump_path = args['rootDir'] + '/HateSpeechData.pkl'
        else:
            self.vocfile = args['rootDir'] + '/voc.txt'
            self.data_dump_path = args['rootDir'] + '/HateSpeechData_noglove.pkl'


        print(self.data_dump_path)
        datasetExist = os.path.isfile(self.data_dump_path)

        if not datasetExist:  # First time we load the database: creating all files
            print('Training data not found. Creating dataset...')
            datasplit = json.load(open(self.corpus_file_split, 'r'))
            data = json.load(open(self.corpus_file, 'r'))
            total_words = []
            dataset = {'train': [], 'val':[], 'test':[]}

            for ss in ['train', 'val', 'test']:
                for id in datasplit[ss]:
                    count={}
                    count['offensive'] = 0 # 0
                    count['hatespeech'] = 0 # 1
                    count['normal'] = 0 #2

                    case = data[id]
                    sen = case['post_tokens']
                    annos = case['annotators']
                    rationales = []
                    for anno in annos:
                        count[anno['label']] += 1

                    if count['offensive'] >= 2:
                        y = 0
                    elif count['hatespeech'] >= 2:
                        y = 1
                    elif count['normal'] >= 2:
                        y = 2
                    else:
                        print('1 1 1 error')
                        exit()

                    rationales = case['rationales']
                    dataset[ss].append([sen, y, rationales])
                    total_words.extend(sen)

            if not use_pretrained_vector:
                fdist = nltk.FreqDist(total_words)
                sort_count = fdist.most_common(30000)
                print('sort_count: ', len(sort_count))
                print(len(set([w for w,c in sort_count])))
                with open(self.vocfile, "w") as v:
                    for w, c in tqdm(sort_count):
                        # if nnn > 0:
                        #     print([(ord(w1),w1) for w1 in w])
                        #     nnn-= 1
                        if w not in [' ', '', '\n', '\r', '\r\n'] and ' ' not in w:
                            v.write(w)
                            v.write(' ')
                            v.write(str(c))
                            v.write('\n')

                    v.close()
                self.word2index = self.read_word2vec(self.vocfile)
                sorted_word_index = sorted(self.word2index.items(), key=lambda item: item[1])
                print('sorted')
                self.index2word = [w for w, n in sorted_word_index]
                print('index2word')
            else:
                self.word2index, self.index2word, self.index2vector = self.read_word2vec_from_pretrained(self.vocfile, 50000)
                print('sorted')
                print('index2word')

            self.index2word_set = set(self.index2word)


            print(len(dataset['train']), len(dataset['val']), len(dataset['test']))


            # self.raw_sentences = copy.deepcopy(dataset)
            for setname in ['train', 'val', 'test']:
                dataset[setname] = [(self.TurnWordID(sen), y, sen, rational) for sen, y, rational in tqdm(dataset[setname])]

            # Saving
            print('Saving dataset...')
            self.saveDataset(self.data_dump_path, dataset, use_pretrained_vector)  # Saving tf samples
        else:
            dataset = self.loadDataset(self.data_dump_path, use_pretrained_vector)
            print('loaded')

        return  dataset

    def saveDataset(self, filename, datasets, use_pretrained_vector):
        """Save samples to file
        Args:
            filename (str): pickle filename
        """
        with open(os.path.join(filename), 'wb') as handle:
            data = {  # Warning: If adding something here, also modifying loadDataset
                'word2index': self.word2index,
                'index2word': self.index2word,
                'datasets': datasets
            }
            if use_pretrained_vector:
                data['index2vec'] = self.index2vector
            pickle.dump(data, handle, -1)  # Using the highest protocol available


    def loadDataset(self, filename, use_pretrained_vector):
        """Load samples from file
        Args:
            filename (str): pickle filename
        """
        dataset_path = os.path.join(filename)
        print('Loading dataset from {}'.format(dataset_path))
        with open(dataset_path, 'rb') as handle:
            data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
            self.word2index = data['word2index']
            self.index2word = data['index2word']
            if use_pretrained_vector:
                self.index2vector = data['index2vec']
            datasets = data['datasets']
        print('training: \t', len(datasets['train']))
        print('dev: \t', len(datasets['val']))
        print('testing: \t', len(datasets['test']))
        self.index2word_set = set(self.index2word)
        print('w2i shape: ', len(self.word2index))
        print('i2w shape: ', len(self.index2word))
        return  datasets

    def read_word2vec(self, vocfile ):
        word2index = dict()
        word2index['[PAD]'] = 0
        word2index['[START_TOKEN]'] = 1
        word2index['[END_TOKEN]'] = 2
        word2index['[UNK]'] = 3
        cnt = 4
        with open(vocfile, "r") as v:

            for line in v:
                word = line.strip().split()[0]
                if word in word2index:
                    sfdg=0
                word2index[word] = cnt
                print(word,cnt)
                cnt += 1

        print(len(word2index),cnt)
        # dic = {w:numpy.random.normal(size=[int(sys.argv[1])]).astype('float32') for w in word2index}
        print ('Dictionary Got!')
        return word2index

    def read_word2vec_from_pretrained(self, embfile, topk_word_num=-1):
        fopen = gzip.open if embfile.endswith(".gz") else open
        word2index = dict()
        word2index['[PAD]'] = 0
        word2index['[START_TOKEN]'] = 1
        word2index['[END_TOKEN]'] = 2
        word2index['[UNK]'] = 3
        # word2index['PAD'] = 1
        # word2index['UNK'] = 0

        cnt = 4
        vectordim = -1
        index2vector = []
        with fopen(embfile, "r") as v:
            lines = v.readlines()
            if topk_word_num > 0:
                lines = lines[:topk_word_num]
            for line in tqdm(lines):

                word_vec = line.strip().split()
                word = word_vec[0]
                # if word=='Dodger':
                #     print(cnt, len(word2index))
                #     exit()

                vector = np.asarray([float(value) for value in word_vec[1:]])
                if vectordim == -1:
                    vectordim = len(vector)
                index2vector.append(vector)
                word2index[word] = cnt
                print(word, cnt)
                cnt += 1
        print('before add special token:' , len(index2vector))
        index2vector = [np.random.normal(size=[vectordim]).astype('float32') for _ in range(4)] + index2vector
        print('after add special token:' ,len(index2vector))
        index2vector = np.asarray(index2vector)
        index2word = [w for w, n in word2index.items()]
        print(len(word2index), cnt)
        print('Dictionary Got!')
        return word2index, index2word, index2vector

    def TurnWordID(self, words):
        res = []
        for w in words:
            w = w.lower()
            if w in self.index2word_set:
                id = self.word2index[w]
                if id == 28039:
                    dfg=0
                res.append(id)
            else:
                res.append(self.word2index['[UNK]'])
        return res


    def printBatch(self, batch):
        """Print a complete batch, useful for debugging
        Args:
            batch (Batch): a batch object
        """
        print('----- Print batch -----')
        for i in range(len(batch.encoderSeqs[0])):  # Batch size
            print('Encoder: {}'.format(self.batchSeq2str(batch.encoderSeqs, seqId=i)))
            print('Decoder: {}'.format(self.batchSeq2str(batch.decoderSeqs, seqId=i)))
            print('Targets: {}'.format(self.batchSeq2str(batch.targetSeqs, seqId=i)))
            print('Weights: {}'.format(' '.join([str(weight) for weight in [batchWeight[i] for batchWeight in batch.weights]])))

    def sequence2str(self, sequence, clean=False, reverse=False):
        """Convert a list of integer into a human readable string
        Args:
            sequence (list<int>): the sentence to print
            clean (Bool): if set, remove the <go>, <pad> and <eos> tokens
            reverse (Bool): for the input, option to restore the standard order
        Return:
            str: the sentence
        """

        if not sequence:
            return ''

        if not clean:
            return ' '.join([self.index2word[idx] for idx in sequence])

        sentence = []
        for wordId in sequence:
            if wordId == self.word2index['[END_TOKEN]']:  # End of generated sentence
                break
            elif wordId != self.word2index['[PAD]'] and wordId != self.word2index['[START_TOKEN]']:
                sentence.append(self.index2word[wordId])

        if reverse:  # Reverse means input so no <eos> (otherwise pb with previous early stop)
            sentence.reverse()

        return self.detokenize(sentence)

    def detokenize(self, tokens):
        """Slightly cleaner version of joining with spaces.
        Args:
            tokens (list<string>): the sentence to print
        Return:
            str: the sentence
        """
        return ''.join([
            ' ' + t if not t.startswith('\'') and
                       t not in string.punctuation
                    else t
            for t in tokens]).strip().capitalize()

    def batchSeq2str(self, batchSeq, seqId=0, **kwargs):
        """Convert a list of integer into a human readable string.
        The difference between the previous function is that on a batch object, the values have been reorganized as
        batch instead of sentence.
        Args:
            batchSeq (list<list<int>>): the sentence(s) to print
            seqId (int): the position of the sequence inside the batch
            kwargs: the formatting options( See sequence2str() )
        Return:
            str: the sentence
        """
        sequence = []
        for i in range(len(batchSeq)):  # Sequence length
            sequence.append(batchSeq[i][seqId])
        return self.sequence2str(sequence, **kwargs)

    def sentence2enco(self, sentence):
        """Encode a sequence and return a batch as an input for the model
        Return:
            Batch: a batch object containing the sentence, or none if something went wrong
        """

        if sentence == '':
            return None

        # First step: Divide the sentence in token
        tokens = nltk.word_tokenize(sentence)
        if len(tokens) > args['maxLength']:
            return None

        # Second step: Convert the token in word ids
        wordIds = []
        for token in tokens:
            wordIds.append(self.getWordId(token, create=False))  # Create the vocabulary and the training sentences

        # Third step: creating the batch (add padding, reverse)
        batch = self._createBatch([[wordIds, []]])  # Mono batch, no target output

        return batch

    def deco2sentence(self, decoderOutputs):
        """Decode the output of the decoder and return a human friendly sentence
        decoderOutputs (list<np.array>):
        """
        sequence = []

        # Choose the words with the highest prediction score
        for out in decoderOutputs:
            sequence.append(np.argmax(out))  # Adding each predicted word ids

        return sequence  # We return the raw sentence. Let the caller do some cleaning eventually

    def playDataset(self):
        """Print a random dialogue from the dataset
        """
        print('Randomly play samples:')
        print(len(self.datasets['train']))
        for i in range(args['playDataset']):
            idSample = random.randint(0, len(self.datasets['train']) - 1)
            print('sen: {} {}'.format(self.sequence2str(self.datasets['train'][idSample][0], clean=True), self.datasets['train'][idSample][1]))
            print()
        pass


def tqdm_wrap(iterable, *args, **kwargs):
    """Forward an iterable eventually wrapped around a tqdm decorator
    The iterable is only wrapped if the iterable contains enough elements
    Args:
        iterable (list): An iterable object which define the __len__ method
        *args, **kwargs: the tqdm parameters
    Return:
        iter: The iterable eventually decorated
    """
    if len(iterable) > 100:
        return tqdm(iterable, *args, **kwargs)
    return iterable
