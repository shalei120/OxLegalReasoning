import functools
print = functools.partial(print, flush=True)
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
        self.encoderSeqs = []
        self.encoder_lens = []
        self.label = []
        self.decoderSeqs = []
        self.targetSeqs = []
        self.decoder_lens = []


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

        self.lawinfo['i2c'] = {i:c for c,i in self.lawinfo['c2i'].items() }
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
            sen_ids, charge_list, law, toi, raw_sentence = samples[i]

            if len(sen_ids) > args['maxLengthEnco']:
                sen_ids = sen_ids[:args['maxLengthEnco']]

            batch.encoderSeqs.append(sen_ids)
            batch.encoder_lens.append(len(batch.encoderSeqs[i]))
            if args['task'] == 'charge':
                batch.label.append(charge_list)
            elif args['task'] == 'law':
                batch.label.append(law)
            elif args['task'] == 'toi':
                batch.label.append(toi)

        maxlen_enc = max(batch.encoder_lens)
        if args['task'] in ['charge', 'law']:
            maxlen_charge = max([len(c) for c in batch.label]) + 1
        # args['chargenum']      eos
        # args['chargenum'] + 1  padding

        for i in range(batchSize):
            batch.encoderSeqs[i] = batch.encoderSeqs[i] + [self.word2index['PAD']] * (maxlen_enc - len(batch.encoderSeqs[i]))
            if args['task'] in ['charge', 'law']:
                batch.label[i] = batch.label[i] + [args['chargenum']] + [args['chargenum']+1] * (maxlen_charge -1 - len(batch.label[i]))

        return batch

    def getBatches(self, setname = 'train'):
        """Prepare the batches for the current epoch
        Return:
            list<Batch>: Get a list of the batches for the next epoch
        """
        if setname not in self.batches:
            self.shuffle()
            if  args['classify_type'] == 'single':
                self.datasets[setname] = [d for d in self.datasets[setname] if len(d[1]) == 1]

            batches = []
            print(len(self.datasets[setname]))
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

    def getChargeNum(self):
        """Return the number of words present in the dataset
        Return:
            int: Number of word on the loader corpus
        """
        return len(self.lawinfo['c2i'])

    def getLawNum(self):
        """Return the number of words present in the dataset
        Return:
            int: Number of word on the loader corpus
        """
        return len(self.lawinfo['law2i'])

    def GetTermImprisonment(self, toi):
        '''
        :param toi: {'life_imprisonment': False, 'death_penalty': False, 'imprisonment': 4}
        :return:
        '''
        map_list = {
            0: "死刑或无期",
            1: "十年以上",
            2: "七到十年",
            3: "五到七年",
            4: "三到五年",
            5: "二到三年",
            6: "一到二年",
            7: "九到十二个月",
            8: "六到九个月",
            9: "零到六个月",
            10: "没事"
        }
        if toi['life_imprisonment'] or toi['death_penalty']:
            return 0
        elif toi['imprisonment'] >= 120:
            return 1
        elif 84 <= toi['imprisonment'] < 120:
            return 2
        elif 60 <= toi['imprisonment'] < 84:
            return 3
        elif 36 <= toi['imprisonment'] < 60:
            return 4
        elif 24 <= toi['imprisonment'] < 36:
            return 5
        elif 12 <= toi['imprisonment'] < 24:
            return 6
        elif 9 <= toi['imprisonment'] < 12:
            return 7
        elif 6 <= toi['imprisonment'] < 9:
            return 8
        elif 0 < toi['imprisonment'] < 6:
            return 9
        elif toi['imprisonment'] == 0 :
            return 10


    def loadCorpus_CAIL(self):
        """Load/create the conversations data
        """
        if args['datasetsize'] == 'big':
            self.basedir = '../Legal/final_all_data/first_stage/'
            self.corpus_file_train = self.basedir + 'train.json'
            self.corpus_file_test =  self.basedir + 'test.json'
            self.data_dump_path = args['rootDir'] + '/CAILdata.pkl'
            self.vocfile = args['rootDir'] + '/voc_big.txt'
        elif args['datasetsize'] == 'small':
            self.basedir = '../Legal/final_all_data/exercise_contest/'
            self.corpus_file_train = self.basedir + 'data_train.json'
            self.corpus_file_test =  self.basedir + 'data_test.json'
            self.data_dump_path = args['rootDir'] + '/CAILdata_small.pkl'
            self.vocfile = args['rootDir'] + '/voc_small.txt'

        print(self.data_dump_path)
        datasetExist = os.path.isfile(self.data_dump_path)

        if not datasetExist:  # First time we load the database: creating all files
            print('Training data not found. Creating dataset...')

            total_words = []
            dataset = {'train': [], 'test':[]}

            ################################################################
            # charge names
            law_related_info = {}
            with open('./accu.txt', 'r') as rh:
                lines = rh.readlines()
                lines = [line.strip() for line in lines]
                charge2index = {c: i for i, c in enumerate(lines)}

            law_related_info['c2i'] = charge2index
            charge2num = {c:0 for c,i in charge2index.items()}


            with open('./law.txt', 'r') as rh:
                lines = rh.readlines()
                lines = [line.strip() for line in lines]
                law2index = {c: i for i, c in enumerate(lines)}

            law_related_info['law2i'] = law2index
            law2num = {c:0 for c,i in law2index.items()}

            ################################################################

            with open(self.corpus_file_train, 'r',encoding="utf-8") as rhandle:
                lines = rhandle.readlines()

                # sentences = []
                for line in tqdm(lines):
                    cases = json.loads(line)
                    fact_text = cases['fact']       # str
                    law_article = cases['meta']['relevant_articles']  # ['23','34']
                    accusation = cases['meta']['accusation']   # ['steal']
                    criminals =  cases['meta']['criminals']    # ['jack']
                    term_of_imprisonment = cases['meta']['term_of_imprisonment']   # {'life_imprisonment': False, 'death_penalty': False, 'imprisonment': 4}
                    punish_of_money = cases['meta']['punish_of_money'] # int 1000

                    if len(accusation) >1 or len(law_article) > 1 or len(criminals) > 1:
                        continue
                    if args['datasetsize'] == 'small':
                        law_article = [str(l) for l in law_article]
                    fact_text = self.tokenizer(fact_text)
                    total_words.extend(fact_text)
                    dataset['train'].append((fact_text, accusation, law_article, self.GetTermImprisonment(term_of_imprisonment)))

                    try:
                        charge2num[accusation[0]] += 1
                    except:
                        accusation[0] = accusation[0].replace('[','').replace(']','')
                        charge2num[accusation[0]] += 1

                    law2num[str(law_article[0])] += 1

            high_freq_charges = [c for c,n in charge2num.items() if n >100]
            high_freq_laws = [c for c,n in law2num.items() if n >100]
            law_related_info['c2i'] = {c:i for i,c in enumerate(high_freq_charges)}
            law_related_info['law2i'] = {c:i for i,c in enumerate(high_freq_laws)}
            dataset['train'] = [it for it in dataset['train'] if it[1][0] in law_related_info['c2i'] and it[2][0] in law_related_info['law2i']]

            with open(self.corpus_file_test, 'r') as rhandle:
                lines = rhandle.readlines()
                # sentences = []
                # charges = []
                for line in tqdm(lines):
                    cases = json.loads(line)

                    fact_text = cases['fact']       # str
                    law_article = cases['meta']['relevant_articles']  # ['23','34']
                    accusation = cases['meta']['accusation']   # ['steal']
                    criminals =  cases['meta']['criminals']    # ['jack']
                    term_of_imprisonment = cases['meta']['term_of_imprisonment']   # {'life_imprisonment': False, 'death_penalty': False, 'imprisonment': 4}
                    punish_of_money = cases['meta']['punish_of_money'] # int 1000

                    if len(accusation) >1 or len(law_article) > 1 or len(criminals) > 1:
                        continue
                    if args['datasetsize'] == 'small':
                        law_article = [str(l) for l in law_article]
                    if accusation[0] not in law_related_info['c2i'] or law_article[0] not in law_related_info['law2i']:
                        continue

                    fact_text = self.tokenizer(fact_text)
                    total_words.extend(fact_text)
                    dataset['test'].append((fact_text, accusation, law_article, self.GetTermImprisonment(term_of_imprisonment)))

            # with open(args['rootDir'] + '/datadump.tmp', 'rb') as handle:
            #     dataset = pickle.load(handle)
            #     print('tmp loaded')
            #     for ft, ac in tqdm(dataset['train']):
            #         total_words.extend(ft)

            print(len(dataset['train']), len(dataset['test']))

            # with open(args['rootDir'] + '/datadump.tmp', 'wb') as handle:
            #     pickle.dump(dataset, handle, -1)
            #     print('tmp stored')

            fdist = nltk.FreqDist(total_words)
            sort_count = fdist.most_common(30000)
            print('sort_count: ', len(sort_count))

            # nnn=8
            with open(self.vocfile, "w") as v:
                for w, c in tqdm(sort_count):
                    # if nnn > 0:
                    #     print([(ord(w1),w1) for w1 in w])
                    #     nnn-= 1
                    if w not in [' ', '', '\n', '\r', '\r\n']:
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
            self.index2word_set = set(self.index2word)

            # self.raw_sentences = copy.deepcopy(dataset)
            for setname in ['train', 'test']:
                dataset[setname] = [(self.TurnWordID(sen), [law_related_info['c2i'][c] for c in charge], [law_related_info['law2i'][c] for c in law],toi, sen) for sen, charge,law, toi in tqdm(dataset[setname])]
            # Saving
            print('Saving dataset...')
            self.saveDataset(self.data_dump_path, dataset, law_related_info)  # Saving tf samples
        else:
            dataset, law_related_info = self.loadDataset(self.data_dump_path)
            print('train size:\t', len(dataset['train']))
            print('test size:\t', len(dataset['test']))
            print('loaded')

        return  dataset, law_related_info

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
                'lawinfo' : law_related_info
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
        return  datasets, law_related_info


    def read_word2vec(self, vocfile ):
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
                print(word,cnt)
                cnt += 1

        print(len(word2index),cnt)
        # dic = {w:numpy.random.normal(size=[int(sys.argv[1])]).astype('float32') for w in word2index}
        print ('Dictionary Got!')
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
