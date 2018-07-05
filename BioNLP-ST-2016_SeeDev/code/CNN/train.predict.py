##########################################################################################################
###                                        train.predict.py                                            ###
##########################################################################################################
'''
This script is to trian CNN model and preidict.
'''

import pickle as pkl


trian_dev_embedding = 'D:/Minzhe.intern/BioNLP/BioNLP-ST-2016_SeeDev/data/train_dev_embedding.pkl'
with open(trian_dev_embedding, 'rb') as f:
    embedding_data = pkl.load(f)

wordEmbeddings = embedding_data['wordEmbeddings']
word2Idx = embedding_data['word2Idx']