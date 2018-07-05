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
from sklearn.preprocessing import LabelEncoder
import pickle as pkl


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
def generateDataMatrix(data_file, word2Idx, min_dist=-30, max_dist=30, sent_len=100):
    '''
    Generate input data matrix.
    '''
    tokenID_mat, e1dist_mat, e2dist_mat, e1pos_mat, e2pos_mat = [], [], [], [], []

    data = pd.read_csv(data_file, sep='\t')
    for idx, row in data.iterrows():
        tokens = row['sent'].split(' ')
        e1_loc = list(map(lambda x: int(x), row['e1_loc'].split(',')))
        e2_loc = list(map(lambda x: int(x), row['e2_loc'].split(',')))
        tokenID_line = np.zeros(sent_len, dtype='int32')    # padding with 0
        e1dist_line = np.repeat(max_dist+1, sent_len)       # padding with max distance + 1
        e2dist_line = np.repeat(max_dist+1, sent_len)       # padding with max distance + 1
        e1pos_line = np.zeros(sent_len, dtype='int32')      # padding with 0
        e2pos_line = np.zeros(sent_len, dtype='int32')      # padding with 0

        # loop through each tokens
        for j in range(min(sent_len, len(tokens))):
            tokenID_line[j] = getWordIdx(token=tokens[j], word2Idx=word2Idx)
            e1dist_line[j] = mapDist(dist=j-e1_loc[0], min_dist=-30, max_dist=30)
            e2dist_line[j] = mapDist(dist=j-e2_loc[0], min_dist=-30, max_dist=30)
            e1pos_line[j] = labelWordLoc(loc=j, entity_locs=e1_loc)
            e2pos_line[j] = labelWordLoc(loc=j, entity_locs=e2_loc)
        
        tokenID_mat.append(tokenID_line)
        e1dist_mat.append(e1dist_line)
        e2dist_mat.append(e2dist_line)
        e1pos_mat.append(e1pos_line)
        e2pos_mat.append(e2pos_line)
    
    return np.array(tokenID_mat, dtype='int32'), \
           np.array(e1dist_mat, dtype='int32'), np.array(e2dist_mat, dtype='int32'), \
           np.array(e1pos_mat, dtype='int32'), np.array(e2pos_mat, dtype='int32')


# >>>>>>>>>>>>>>>>>> utility <<<<<<<<<<<<<<<<< #
def getWordIdx(token, word2Idx): 
    '''
    Returns from the word2Idx table the word index for a given token
    '''      
    if token in word2Idx.keys():
        return word2Idx[token]
    elif token.lower() in word2Idx.keys():
        return word2Idx[token.lower()]
    else:
        return word2Idx["UNKNOWN_TOKEN"]

def mapDist(dist, min_dist=-30, max_dist=30):
    '''
    Measure distance of all words to flag word.
    '''
    if dist < min_dist:
        return min_dist
    elif dist > max_dist:
        return max_dist
    else:
        return dist

def labelWordLoc(loc, entity_locs):
    '''
    Label entity and outer, inner-entity location.
    '''
    if loc < min(entity_locs) or loc > max(entity_locs):
        return -1
    elif loc in entity_locs:
        return 1
    else:
        return 2


################################  read train and dev data  ###################################
train_file = "D:/Minzhe.intern/BioNLP/BioNLP-ST-2016_SeeDev/data/train_relent.txt"
train_data = pd.read_csv(train_file, sep='\t')
train_labels = train_data.rel
train_words = set()
train_max_len = 0
for tokens in train_data.sent.str.split(' '):
    train_max_len = max(train_max_len, len(tokens))
    train_words = train_words | set(tokens)

dev_file = "D:/Minzhe.intern/BioNLP/BioNLP-ST-2016_SeeDev/data/dev_relent.txt"
dev_data = pd.read_csv(dev_file, sep='\t')
dev_labels = dev_data.rel
dev_words = set()
dev_max_len = 0
for tokens in dev_data.sent.str.split(' '):
    dev_max_len = max(dev_max_len, len(tokens))
    dev_words = dev_words | set(tokens)

# --------- collect all appeared words ----------- #
words = train_words | dev_words
sent_max_len = max(train_max_len, dev_max_len)
labels = set(train_labels) | set(dev_labels)

label_encoder = LabelEncoder()
label_encoder.fit(list(labels))
y_train = label_encoder.transform(train_labels)
y_test = label_encoder.transform(dev_labels)

print('Total number of different words: {}'.format(len(words)))
print('Total number of different labels: {}'.format(len(labels)))
print('Training instense: {}'.format(len(y_train)))
print('Testing instense: {}'.format(len(y_test)))
print('Max sentense length: {}\n'.format(sent_max_len))

################################  word embedding  ###################################
# model = KeyedVectors.load_word2vec_format('/home/t-mizha/data/embeddings/bio_nlp_vec/PubMed-shuffle-win-2.bin', binary=True)
# model.save_word2vec_format('/home/t-mizha/data/embeddings/bio_nlp_vec/PubMed-shuffle-win-2.txt', binary=False)
# ----------- create word embeddings ---------------- #
embeddings_path = 'D:/Minzhe.intern/BioNLP/data/embedding/PubMed-shuffle-win-2.txt'
trian_dev_embedding = 'D:/Minzhe.intern/BioNLP/BioNLP-ST-2016_SeeDev/data/train_dev_embedding.pkl'
# createEmbedding(embeddings_path=embeddings_path, words=words, out_path=trian_dev_embedding)

with open(trian_dev_embedding, 'rb') as f:
    embedding_data = pkl.load(f)

wordEmbeddings = embedding_data['wordEmbeddings']
word2Idx = embedding_data['word2Idx']

################################  input data matrix  ###################################
word_train, e1dist_train, e2dist_train, e1pos_train, e2pos_train = generateDataMatrix(data_file=train_file, word2Idx=word2Idx)
word_test, e1dist_test, e2dist_test, e1pos_test, e2pos_test = generateDataMatrix(data_file=dev_file, word2Idx=word2Idx)

print(word_test.shape)
print(word_test[133])
print(e1dist_test.shape)
print(e1dist_test[133])
print(e2dist_test.shape)
print(e2dist_test[133])
print(e1pos_test.shape)
print(e1pos_test[133])
print(e2pos_test.shape)
print(e2pos_test[133])
print(y_train[:1000])
print(y_train[1000:])
print(y_test)