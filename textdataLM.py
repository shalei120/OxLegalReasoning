import numpy as np
import nltk  # For tokenize
from tqdm import tqdm  # Progress bar
import pickle  # Saving the data
import math  # For float comparison
import os  # Checking file existance
import random
import string, copy
from nltk.tokenize import word_tokenize
import jieba
import json
from Hyperparameters import args


class Batch:
    """Struct containing batches info
    """

    def __init__(self):
        self.decoderSeqs = []
        self.targetSeqs = []
        self.decoder_lens = []
        self.raw = []


class TextData:
    """Dataset class
    Warning: No vocabulary limit
    """

    def __init__(self, corpusname):
        """Load all conversations
        Args:
            args: parameters of the model
        """

        # Path variables
        if corpusname == 'cail':
            self.tokenizer = lambda x: list(jieba.cut(x))
        elif corpusname == 'caselaw':
            self.tokenizer = word_tokenize

        self.trainingSamples = []  # 2d array containing each question and his answer [[input,target]]

        self.datasets, self.lawinfo = self.loadCorpus_CAIL()

        self.lawinfo['i2c'] = {i: c for c, i in self.lawinfo['c2i'].items()}
        print(self.lawinfo['i2c'])

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
            sen_ids = samples[i]

            if len(sen_ids) > args['maxLengthEnco']:
                sen_ids = sen_ids[:args['maxLengthEnco']]

            batch.decoderSeqs.append([self.word2index['START_TOKEN']] + sen_ids)
            batch.decoder_lens.append(len(batch.decoderSeqs[i]))
            batch.targetSeqs.append(sen_ids + [self.word2index['END_TOKEN']])
            batch.raw.append([self.index2word[wid] for wid in sen_ids])

        maxlen_dec = max(batch.decoder_lens)
        maxlen_dec = min(maxlen_dec, args['maxLengthEnco'])

        for i in range(batchSize):
            batch.decoderSeqs[i] = batch.decoderSeqs[i] + [self.word2index['PAD']] * (
                        maxlen_dec - len(batch.decoderSeqs[i]))
            batch.targetSeqs[i] = batch.targetSeqs[i] + [self.word2index['PAD']] * (
                        maxlen_dec - len(batch.targetSeqs[i]))

        return batch

    def paragraph2sentence(self, doclist):
        split_tokens = [self.word2index['。'], self.word2index['；'], self.word2index['，']]
        sen_list = []
        count = 0
        cc = 0
        for sen_ids, charge_list, law, toi, raw_sentence in doclist:
            start = 0
            sen_ids = [w for w in sen_ids if w not in ['\r\n']]
            for ind, w in enumerate(sen_ids):
                if w in split_tokens:
                    if 8 < ind - start < 500:
                        sen_list.append(sen_ids[start:ind + 1])
                    # if len(sen_list[-1]) > 500:
                    #     print(raw_sentence[start:ind+1])
                    #     print(sen_list[-1])
                    #     count += 1
                    # if len(sen_list[-1]) <10:
                    #     cc += 1
                    start = ind + 1

            if start < len(sen_ids) - 1:
                sen_list.append(sen_ids[start:])

        # print(len(sen_list),count,cc)
        # exit(0)
        return sen_list

    def getBatches(self, setname='train'):
        """Prepare the batches for the current epoch
        Return:
            list<Batch>: Get a list of the batches for the next epoch
        """
        if setname not in self.batches:
            self.shuffle()
            batches = []
            print(len(self.datasets[setname]))
            dataset_sen = self.paragraph2sentence(self.datasets[setname])
            datanum = len(dataset_sen)
            print(datanum)

            def genNextSamples():
                """ Generator over the mini-batch training samples
                """
                for i in range(0, datanum, args['batchSize']):
                    yield dataset_sen[i:min(i + args['batchSize'], datanum)]

            # TODO: Should replace that by generator (better: by tf.queue)

            for index, samples in enumerate(genNextSamples()):
                # print([self.index2word[id] for id in samples[5][0]], samples[5][2])
                batch = self._createBatch(samples)
                batches.append(batch)

            self.batches[setname] = batches

        # print([self.index2word[id] for id in batches[2].encoderSeqs[5]], batches[2].raws[5])
        return self.batches[setname]

    def getVocabularySize(self):
        """Return the number of words present in the dataset
        Return:
            int: Number of word on the loader corpus
        """
        return len(self.word2index)

    def getChargeNum(self):
        """Return the number of words present in the dataset
        Return:
            int: Number of word on the loader corpus
        """
        return len(self.lawinfo['c2i'])

    def loadCorpus_CAIL(self):
        """Load/create the conversations data
        """
        if args['tasksize'] == 'big':
            self.basedir = '../Legal/final_all_data/first_stage/'
            self.corpus_file_train = self.basedir + 'train.json'
            self.corpus_file_test = self.basedir + 'test.json'
            self.data_dump_path = args['rootDir'] + '/CAILdata.pkl'
        elif args['tasksize'] == 'small':
            self.basedir = '../Legal/final_all_data/exercise_contest/'
            self.corpus_file_train = self.basedir + 'data_train.json'
            self.corpus_file_test = self.basedir + 'data_test.json'

            self.data_dump_path = args['rootDir'] + '/CAILdata_small.pkl'

        print(self.data_dump_path)

        dataset, law_related_info = self.loadDataset(self.data_dump_path)
        print('loaded')

        return dataset, law_related_info

    def saveDataset(self, filename, datasets, law_related_info):
        """Save samples to file
        Args:
            filename (str): pickle filename
        """
        with open(os.path.join(filename), 'wb') as handle:
            data = {  # Warning: If adding something here, also modifying loadDataset
                'word2index': self.word2index,
                'index2word': self.index2word,
                'datasets': datasets,
                'lawinfo': law_related_info
            }
            pickle.dump(data, handle, -1)  # Using the highest protocol available

    def loadDataset(self, filename):
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
            datasets = data['datasets']
            law_related_info = data['lawinfo']

        self.index2word_set = set(self.index2word)
        return datasets, law_related_info

    def read_word2vec(self, vocfile):
        word2index = dict()
        word2index['PAD'] = 0
        word2index['START_TOKEN'] = 1
        word2index['END_TOKEN'] = 2
        word2index['UNK'] = 3
        cnt = 4
        with open(vocfile, "r") as v:
            for line in v:
                word = line.strip().split()[0]
                word2index[word] = cnt
                print(word, cnt)
                cnt += 1

        print(len(word2index), cnt)
        # dic = {w:numpy.random.normal(size=[int(sys.argv[1])]).astype('float32') for w in word2index}
        print('Dictionary Got!')
        return word2index

    def TurnWordID(self, words):
        res = []
        for w in words:
            w = w.lower()
            if w in self.index2word_set:
                id = self.word2index[w]
                res.append(id)
            else:
                res.append(self.word2index['UNK'])
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
            print('Weights: {}'.format(
                ' '.join([str(weight) for weight in [batchWeight[i] for batchWeight in batch.weights]])))

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
            if wordId == self.word2index['END_TOKEN']:  # End of generated sentence
                break
            elif wordId != self.word2index['PAD'] and wordId != self.word2index['START_TOKEN']:
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
            print('sen: {} {}'.format(self.sequence2str(self.datasets['train'][idSample][0], clean=True),
                                      self.datasets['train'][idSample][1]))
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
