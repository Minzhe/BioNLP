##########################################################################################################
###                                         dataLoader.py                                              ###
##########################################################################################################
'''
The file load the processed train and dev table, and convert to matrix.
'''

from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import pandas as pd
import gzip
import pickle as pkl
from sklearn.preprocessing import LabelEncoder
from utility import hLabelEncoder
from utility import encodeMapper 


################################  function  ###################################
# >>>>>>>>>>>>>>>>>> embeddings <<<<<<<<<<<<<<<<< #
def createEmbedding(embeddings_path, words, out_path):
    '''
    Load pre trained word embedding.
    '''
    status = False
    word2Idx = {}
    wordEmbeddings = []
    wordEmbeddings_back = {}
    words_not_found = words

    # ----------- Load the pre-trained embeddings file ------------ #
    with open(embeddings_path, "r", encoding='utf-8') as fEmbeddings:
        print("Loading pre-trained embeddings file ...")
        for idx, line in enumerate(fEmbeddings):
            if idx == 0:
                continue
            line_split = line.strip().split(' ')
            word = line_split[0]

            if len(word2Idx) == 0: # Add padding + unknown
                word2Idx["PADDING_TOKEN"] = len(word2Idx)
                wordEmbeddings.append(np.zeros(len(line_split)-1)) # Zero vector vor 'PADDING' word
                word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
                wordEmbeddings.append(np.random.uniform(-0.25, 0.25, len(line_split)-1))

            # select words appears in training and testing data
            if word in words:
                wordEmbeddings.append(np.array([float(num) for num in line_split[1:]]))
                word2Idx[word] = len(word2Idx)
                words_not_found.remove(word)
            elif word.lower() in list(map(lambda x: x.lower(), words_not_found)):
                wordEmbeddings_back[word] = np.array([float(num) for num in line_split[1:]])
            
    # check for unfound words
    back_words = wordEmbeddings_back.keys()
    for unfound_word in words_not_found:
        for word in back_words:
            if unfound_word.lower() == word.lower():
                wordEmbeddings.append(wordEmbeddings_back[word])
                word2Idx[unfound_word] = len(word2Idx)
                continue

    wordEmbeddings = np.array(wordEmbeddings)
    print("Embeddings shape: {}".format(wordEmbeddings.shape))
    with open(out_path, 'wb') as pkl_f:
        pkl.dump({'wordEmbeddings': wordEmbeddings, 'word2Idx': word2Idx}, file=pkl_f)
    print("Data stored in {}.".format(out_path))
    status = True

    return status

# >>>>>>>>>>>>>>>>>> create data matrix <<<<<<<<<<<<<<<<< #
def generateDataMatrix(data_file, word2Idx, label_encoder, type_encoder, min_dist=-30, max_dist=30, sent_len=100):
    '''
    Generate input data matrix.
    '''
    em = encodeMapper()
    data = pd.read_csv(data_file, sep='\t')
    labels = label_encoder.label2class(data.rel)
    hierarchy = label_encoder.label2hclass(data.rel)
    e1type = type_encoder.transform(data.e1_type)
    e2type = type_encoder.transform(data.e2_type)
    tokenID_mat, e1dist_mat, e2dist_mat, e1type_mat, e2type_mat = [], [], [], [], []

    for idx, row in data.iterrows():
        tokens = row['sent'].split(' ')
        e1_loc = list(map(lambda x: int(x), row['e1_loc'].split(',')))
        e2_loc = list(map(lambda x: int(x), row['e2_loc'].split(',')))
        tokenID_line = np.zeros(sent_len, dtype='int32')    # padding with 0
        e1dist_line = np.repeat(99, sent_len)       # padding with 99
        e2dist_line = np.repeat(99, sent_len)       # padding with 99
        e1type_line = np.zeros(sent_len, dtype='int32')      # padding with 0
        e2type_line = np.zeros(sent_len, dtype='int32')      # padding with 0

        # loop through each tokens
        for j in range(min(sent_len, len(tokens))):
            tokenID_line[j] = em.mapWordIdx(token=tokens[j], word2Idx=word2Idx)
            e1dist_line[j] = em.mapDist(dist=j-e1_loc[0], min_dist=-30, max_dist=30)
            e2dist_line[j] = em.mapDist(dist=j-e2_loc[0], min_dist=-30, max_dist=30)
            e1type_line[j] = em.mapType(loc=j, entity_locs=e1_loc, entity_type=e1type[idx])
            e2type_line[j] = em.mapType(loc=j, entity_locs=e2_loc, entity_type=e2type[idx])
        
        tokenID_mat.append(tokenID_line)
        e1dist_mat.append(e1dist_line)
        e2dist_mat.append(e2dist_line)
        e1type_mat.append(e1type_line)
        e2type_mat.append(e2type_line)
    
    return labels, hierarchy, np.array(tokenID_mat, dtype='int32'), \
           np.array(e1dist_mat, dtype='int32'), np.array(e2dist_mat, dtype='int32'), \
           np.array(e1type_mat, dtype='int32'), np.array(e2type_mat, dtype='int32')

# >>>>>>>>>>>>>>>>>> data matrix for model parameters <<<<<<<<<<<<<<<<< #
def getParam(data_files):
    '''
    This is to get the data parameters for later model construction.
    '''
    words, labels, etypes = set(), set(), set()
    sent_max_len = 0
    dims = []
    for file_ in data_files:
        data_mat = pd.read_csv(file_, sep='\t')
        dims.append(data_mat.shape[0])
        labels = labels | set(data_mat.rel)
        etypes = etypes | set(data_mat.e1_type) | set(data_mat.e2_type)
        for tokens in data_mat.sent.str.split(' '):
            sent_max_len = max(sent_max_len, len(tokens))
            words = words | set(tokens)
    
    return words, labels, etypes, sent_max_len, dims
        

################################  read train and dev data  ###################################
train_file = "/home/t-mizha/project/BioNLP/BioNLP-ST-2016_SeeDev/data/train_relent.txt"
dev_file = "/home/t-mizha/project/BioNLP/BioNLP-ST-2016_SeeDev/data/dev_relent.txt"
words, labels, etypes, sent_max_len, dims = getParam([train_file, dev_file])

# --------------  encode labels  ---------------- #
label_encoder = hLabelEncoder()
type_encoder = LabelEncoder()
type_encoder.fit(list(etypes))

n_labels = len(labels)
n_hierarchy = len(np.unique(label_encoder.label2hclass(labels)))

print('Total number of different words: {}'.format(len(words)))
print('Total number of different labels: {}'.format(n_labels))
print('Total number of different hierarchy: {}'.format(n_hierarchy))
print('Total number of different entity types: {}'.format(len(etypes)))
print('Training instense: {}'.format(dims[0]))
print('Testing instense: {}'.format(dims[1]))
print('Max sentense length: {}\n'.format(sent_max_len))


################################  word embedding  ###################################
# model = KeyedVectors.load_word2vec_format('/home/t-mizha/data/embeddings/bio_nlp_vec/PubMed-shuffle-win-2.bin', binary=True)
# model.save_word2vec_format('/home/t-mizha/data/embeddings/bio_nlp_vec/PubMed-shuffle-win-2.txt', binary=False)
# ----------- create word embeddings ---------------- #
print('Creating word embedding ...')
embeddings_path = '/home/t-mizha/data/embeddings/bio_nlp_vec/PubMed-shuffle-win-2.txt'
trian_dev_embedding = '/home/t-mizha/project/BioNLP/BioNLP-ST-2016_SeeDev/data/train_dev_embedding.pkl'
# createEmbedding(embeddings_path=embeddings_path, words=words, out_path=trian_dev_embedding)

with open(trian_dev_embedding, 'rb') as f:
    embedding_data = pkl.load(f)

wordEmbeddings = embedding_data['wordEmbeddings']
word2Idx = embedding_data['word2Idx']

# -----------  input data matrix  ---------------- #
print('\nCreating positional embedding ...')
sent_len = min(100, sent_max_len)
y1_train, y2_train, word_train, e1dist_train, e2dist_train, e1type_train, e2type_train = \
    generateDataMatrix(data_file=train_file, word2Idx=word2Idx, sent_len=sent_len, label_encoder=label_encoder, type_encoder=type_encoder)
y1_test, y2_test, word_test, e1dist_test, e2dist_test, e1type_test, e2type_test = \
    generateDataMatrix(data_file=dev_file, word2Idx=word2Idx, sent_len=sent_len, label_encoder=label_encoder, type_encoder=type_encoder)

train_dev_mat = '/home/t-mizha/project/BioNLP/BioNLP-ST-2016_SeeDev/data/train_dev_matrix.pkl'
with open(train_dev_mat, 'wb') as pkl_f:
    pkl.dump({'train_data': (y1_train, y2_train, word_train, e1dist_train, e2dist_train, e1type_train, e2type_train),
              'test_data': (y1_test, y2_test, word_test, e1dist_test, e2dist_test, e1type_test, e2type_test),
              'label_encoder': label_encoder,
              'param': (n_labels, n_hierarchy, sent_len)}, file=pkl_f)
print('\nTraining and test data matrix stored in {}'.format(train_dev_mat))