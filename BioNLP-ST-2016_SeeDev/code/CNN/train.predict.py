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
def calMetrics(y_pred, y_true, label_encoder, y_train):
    acc = round(accuracy_score(y_pred=y_pred, y_true=y_true), 3)
    prec = calPrec(y_pred=y_pred, y_true=y_true)
    recall = calRecall(y_pred=y_pred, y_true=y_true)
    f1 = calF1(y_pred=y_pred, y_true=y_true)
    # get labels
    classes = np.unique(np.concatenate([y_pred, y_true]))
    classes = np.sort(classes)
    count_test = np.vectorize(lambda x: np.sum(y_true == x))
    count_train = np.vectorize(lambda x: np.sum(y_train == x))
    class_counts_test = count_test(classes)
    class_counts_train = count_train(classes)
    labels = label_encoder.inverse_transform(classes)
    # table
    metrics_table = pd.DataFrame({'class':classes, 'label':labels, 'train_count':class_counts_train, 'test_count':class_counts_test, 'precision':prec[3], 'recall':recall[3], 'f1':f1[3]})
    metrics_table = metrics_table.append(pd.DataFrame({'class':None, 'label':'Micro', 'train_count':None, 'test_count':None, 'precision':prec[0], 'recall':recall[0], 'f1':f1[0]}, index=[len(metrics_table)+1]))
    metrics_table = metrics_table.append(pd.DataFrame({'class':None, 'label':'Macro', 'train_count':None, 'test_count':None, 'precision':prec[1], 'recall':recall[1], 'f1':f1[1]}, index=[len(metrics_table)+1]))
    metrics_table = metrics_table.append(pd.DataFrame({'class':None, 'label':'Weighted', 'train_count':None, 'test_count':None, 'precision':prec[2], 'recall':recall[2], 'f1':f1[2]}, index=[len(metrics_table)+1]))
    metrics_table = metrics_table.append(pd.DataFrame({'class':None, 'label':'Accuracy', 'train_count':None, 'test_count':None, 'precision':acc, 'recall':None, 'f1':None}, index=[len(metrics_table)+1]))
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
y_train, word_train, e1dist_train, e2dist_train, e1type_train, e2type_train = train_dev_data['train_data']
y_test, word_test, e1dist_test, e2dist_test, e1type_test, e2type_test = train_dev_data['test_data']
label_encoder = train_dev_data['label_encoder']
n_label, sent_length = train_dev_data['param']

#######################  CNN model  #############################
save_path = '/home/t-mizha/project/BioNLP/BioNLP-ST-2016_SeeDev/data/cnnmodel.type.h5'
result_path = '/home/t-mizha/project/BioNLP/BioNLP-ST-2016_SeeDev/result/cnnmodel.type.csv'
batch_size = 64
epochs = 500
learning_rate = 1e-4
dropout = 0.25
validation_split = 0.15
dist_dim = len(np.unique(np.concatenate([e1dist_train, e2dist_train, e1dist_test, e2dist_test])))
type_dim = len(np.unique(np.concatenate([e1type_train, e2type_train, e1type_test, e2type_test])))

# print(y_train[:100])
# print(word_train[:100])
# print(e1dist_train[:100])
# print(e2dist_train[:100])
# print(e1type_train[0])
# print(e2type_train[0])
# print(e1type_train[1])
# print(e2type_train[1])
# print(dist_dim, type_dim)
# exit()

cnnmodel = cnn(embeddings=wordEmbeddings, n_label=n_label, sent_length=sent_length, 
               indist_dim=dist_dim, intype_dim=type_dim, outdist_dim=32, outtype_dim=8,
               learning_rate=learning_rate, dropout=dropout)
cnnmodel.train(X_train=[word_train, e1dist_train, e2dist_train, e1type_train, e2type_train], 
               y_train=y_train, 
               validation_split=validation_split, 
               save_path=save_path, 
               batch_size=batch_size, 
               epochs=epochs)
cnnmodel.loadModel(save_path)

###################  evaluation  ######################
pred_test = cnnmodel.predict_calss(X=[word_test, e1dist_test, e2dist_test, e1type_test, e2type_test])
acc = accuracy_score(y_true=y_test, y_pred=pred_test)
metrics_table = calMetrics(y_pred=pred_test, y_true=y_test, label_encoder=label_encoder, y_train=y_train)
metrics_table.to_csv(result_path, index=False)
print(metrics_table)
print('Accuracy: {0:.3f}'.format(acc))


