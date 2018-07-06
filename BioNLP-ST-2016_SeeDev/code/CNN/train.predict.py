##########################################################################################################
###                                        train.predict.py                                            ###
##########################################################################################################
'''
This script is to trian CNN model and preidict.
'''
import numpy as np
import pandas as pd
import pickle as pkl
from cnn import cnn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

############################  function  #################################
def calMetrics(y_pred, y_true, label_encoder):
    acc = round(accuracy_score(y_pred=y_pred, y_true=y_true), 3)
    prec = calPrec(y_pred=y_pred, y_true=y_true)
    recall = calRecall(y_pred=y_pred, y_true=y_true)
    f1 = calF1(y_pred=y_pred, y_true=y_true)
    # get labels
    classes = np.unique(np.concatenate([y_pred, y_true]))
    classes = np.sort(classes)
    count_val = np.vectorize(lambda x: np.sum(y_true == x))
    class_counts = count_val(classes)
    labels = label_encoder.inverse_transform(classes)
    # table
    metrics_table = pd.DataFrame({'class':classes, 'label':labels, 'count':class_counts, 'precision':prec[3], 'recall':recall[3], 'f1':f1[3]})
    metrics_table = metrics_table.append(pd.DataFrame({'class':None, 'label':'Micro', 'count':None, 'precision':prec[0], 'recall':recall[0], 'f1':f1[0]}, index=[len(metrics_table)+1]))
    metrics_table = metrics_table.append(pd.DataFrame({'class':None, 'label':'Macro', 'count':None, 'precision':prec[1], 'recall':recall[1], 'f1':f1[1]}, index=[len(metrics_table)+1]))
    metrics_table = metrics_table.append(pd.DataFrame({'class':None, 'label':'Weighted', 'count':None, 'precision':prec[2], 'recall':recall[2], 'f1':f1[2]}, index=[len(metrics_table)+1]))
    metrics_table = metrics_table.append(pd.DataFrame({'class':None, 'label':'Accuracy', 'count':None, 'precision':acc, 'recall':None, 'f1':None}, index=[len(metrics_table)+1]))
    return metrics_table

def calPrec(y_pred, y_true):
    prec_micro = precision_score(y_true=y_true, y_pred=y_pred, average='micro')
    prec_macro = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
    prec_weighted = precision_score(y_true=y_true, y_pred=y_pred, average='weighted')
    prec_all = precision_score(y_true=y_true, y_pred=y_pred, average=None)
    return round(prec_micro, 3), round(prec_macro, 3), round(prec_weighted, 3), np.around(prec_all, 3)

def calRecall(y_pred, y_true):
    recall_micro = recall_score(y_true=y_true, y_pred=y_pred, average='micro')
    recall_macro = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
    recall_weighted = recall_score(y_true=y_true, y_pred=y_pred, average='weighted')
    recall_all = recall_score(y_true=y_true, y_pred=y_pred, average=None)
    return round(recall_micro, 3), round(recall_macro, 3), round(recall_weighted, 3), np.around(recall_all,3)

def calF1(y_pred, y_true):
    f1_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1_weighted = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    f1_all = f1_score(y_true=y_true, y_pred=y_pred, average=None)
    return round(f1_micro, 3), round(f1_macro, 3), round(f1_weighted, 3), np.around(f1_all, 3)



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
label_encoder = train_dev_data['label_encoder']
n_label, sent_length = train_dev_data['param']

#######################  CNN model  #############################
save_path = '/home/t-mizha/project/BioNLP/BioNLP-ST-2016_SeeDev/data/cnnmodel.h5'
result_path = '/home/t-mizha/project/BioNLP/BioNLP-ST-2016_SeeDev/result/cnnmodel.csv'
batch_size = 64
epochs = 500
learning_rate = 1e-4
dropout = 0.25
validation_split = 0.15
dist_dim = len(np.unique(np.concatenate([e1dist_train, e2dist_train, e1dist_test, e2dist_test])))
pos_dim = len(np.unique(np.concatenate([e1pos_train, e2pos_train, e1pos_test, e2pos_test])))

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
metrics_table = calMetrics(y_pred=pred_test, y_true=y_test, label_encoder=label_encoder)
metrics_table.to_csv(result_path, index=False)
print(metrics_table)
print('Accuracy: {0:.3f}'.format(acc))


