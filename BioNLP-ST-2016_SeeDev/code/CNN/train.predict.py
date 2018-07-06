##########################################################################################################
###                                        train.predict.py                                            ###
##########################################################################################################
'''
This script is to trian CNN model and preidict.
'''
import numpy as np
import pickle as pkl
from cnn import cnn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calPrec(y_pred, y_true):
    prec_micro = precision_score(y_true=y_true, y_pred=y_pred, average='micro')
    prec_macro = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
    prec_weighted = precision_score(y_true=y_true, y_pred=y_pred, average='weighted')
    prec_all = precision_score(y_true=y_true, y_pred=y_pred, average=None)
    return prec_micro, prec_macro, prec_weighted, prec_all

def calRecall(y_pred, y_true):
    recall_micro = recall_score(y_true=y_true, y_pred=y_pred, average='micro')
    recall_macro = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
    recall_weighted = recall_score(y_true=y_true, y_pred=y_pred, average='weighted')
    recall_all = recall_score(y_true=y_true, y_pred=y_pred, average=None)
    return recall_micro, recall_macro, recall_weighted, recall_all

def calF1(y_pred, y_true):
    f1_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1_weighted = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    f1_all = f1_score(y_true=y_true, y_pred=y_pred, average=None)
    return f1_micro, f1_macro, f1_weighted, f1_all


#################################  load data  #####################################
trian_dev_embedding = '/home/t-mizha/project/BioNLP/BioNLP-ST-2016_SeeDev/data/train_dev_embedding.pkl'
with open(trian_dev_embedding, 'rb') as f:
    embedding_data = pkl.load(f)
wordEmbeddings = embedding_data['wordEmbeddings']
word2Idx = embedding_data['word2Idx']

train_dev_mat = '/home/t-mizha/project/BioNLP/BioNLP-ST-2016_SeeDev/data/train_dev_matrix.pkl'
with open(train_dev_mat, 'rb') as f:
    train_dev_data = pkl.load(f)
word_train, e1dist_train, e2dist_train, e1pos_train, e2pos_train = train_dev_data['X_train']
word_test, e1dist_test, e2dist_test, e1pos_test, e2pos_test = train_dev_data['X_test']
y_train = train_dev_data['y_train']
y_test = train_dev_data['y_test']
n_label, sent_length = train_dev_data['param']

#######################  CNN model  #############################
save_path = '/home/t-mizha/project/BioNLP/BioNLP-ST-2016_SeeDev/data/cnnmodel.h5'
batch_size = 64
epochs = 500
learning_rate = 1e-4
dropout = 0.25
validation_split = 0.15
dist_dim = len(np.unique(np.concatenate([e1dist_train, e2dist_train, e1dist_test, e2dist_test])))
pos_dim = len(np.unique(np.concatenate([e1pos_train, e2pos_train, e1pos_test, e2pos_test])))
print(dist_dim, pos_dim, n_label, sent_length)

cnnmodel = cnn(embeddings=wordEmbeddings, n_label=n_label, sent_length=sent_length, indist_dim=dist_dim, inpos_dim=pos_dim, 
               outdist_dim=32, outpos_dim=4, learning_rate=learning_rate, dropout=dropout)
# cnnmodel.train(X_train=[word_train, e1dist_train, e2dist_train, e1pos_train, e2pos_train], y_train=y_train, 
#                validation_split=0.1, 
#                save_path=save_path, 
#                batch_size=batch_size, 
#                epochs=epochs)
cnnmodel.loadModel(save_path)

###################  evaluation  ######################
pred_test = cnnmodel.predict_calss(X=[word_test, e1dist_test, e2dist_test, e1pos_test, e2pos_test])
acc = accuracy_score(y_true=y_test, y_pred=pred_test)
prec = calPrec(y_pred=pred_test, y_true=y_test)
recall = calRecall(y_pred=pred_test, y_true=y_test)
f1 = calF1(y_pred=pred_test, y_true=y_test)
print('Accuracy: {0:.3f}'.format(acc))
print('Micro precision: {0:.3f}\trecall: {1:.3f}\tf1: {2:.3f}'.format(prec[0], recall[0], f1[0]))
print('Macro precision: {0:.3f}\trecall: {1:.3f}\tf1: {2:.3f}'.format(prec[1], recall[1], f1[1]))
print('Weighted precision: {0:.3f}\trecall: {1:.3f}\tf1: {2:.3f}'.format(prec[2], recall[2], f1[2]))
