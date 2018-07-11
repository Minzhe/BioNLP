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
from utility import calMetrics
from utility import dataSampler
pd.set_option('display.max_columns', 500)


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
save_path = '/home/t-mizha/project/BioNLP/BioNLP-ST-2016_SeeDev/data/cnnmodel.type.duplicate.h5'
result_path = '/home/t-mizha/project/BioNLP/BioNLP-ST-2016_SeeDev/result/cnnmodel.type.duplicate.csv'
batch_size = 64
epochs = 150
learning_rate = 1e-4
dropout = 0.25
validation_split = 0.15
dist_dim = len(np.unique(np.concatenate([e1dist_train, e2dist_train, e1dist_test, e2dist_test])))
type_dim = len(np.unique(np.concatenate([e1type_train, e2type_train, e1type_test, e2type_test])))

cnnmodel = cnn(embeddings=wordEmbeddings, n_label=n_label, sent_length=sent_length, 
               indist_dim=dist_dim, intype_dim=type_dim, outdist_dim=32, outtype_dim=4,
               learning_rate=learning_rate, dropout=dropout)
data_sampler = dataSampler(y=y_train, target_n=400)
for iepoch in range(epochs):
    print('epoch {} ...............'.format(iepoch))
    idx = data_sampler.sample()
    cnnmodel.train(X_train=[word_train[idx], e1dist_train[idx], e2dist_train[idx], e1type_train[idx], e2type_train[idx]], 
                    y_train=y_train[idx], 
                    validation_split=validation_split, 
                    save_path=save_path, 
                    batch_size=100, 
                    epochs=5, verbose=1)
cnnmodel.loadModel(save_path)

###################  evaluation  ######################
pred_test = cnnmodel.predict_calss(X=[word_test, e1dist_test, e2dist_test, e1type_test, e2type_test])
metrics_table = calMetrics(y_pred=pred_test, y_true=y_test, label_encoder=label_encoder, y_train=y_train)
metrics_table.to_csv(result_path, index=False)
print(metrics_table)
# idx = y_test == 1
# y = cnnmodel.predict_calss(X=[word_test[idx], e1dist_test[idx], e2dist_test[idx], e1type_test[idx], e2type_test[idx]])
# print(list(zip(y_test[idx], y)))
# idx = y_test == 2
# y = cnnmodel.predict_calss(X=[word_test[idx], e1dist_test[idx], e2dist_test[idx], e1type_test[idx], e2type_test[idx]])
# print(list(zip(y_test[idx], y)))
# idx = y_test == 3
# y = cnnmodel.predict_calss(X=[word_test[idx], e1dist_test[idx], e2dist_test[idx], e1type_test[idx], e2type_test[idx]])
# print(list(zip(y_test[idx], y)))
# idx = y_test == 5
# y = cnnmodel.predict_calss(X=[word_test[idx], e1dist_test[idx], e2dist_test[idx], e1type_test[idx], e2type_test[idx]])
# print(list(zip(y_test[idx], y)))
# idx = y_test == 6
# y = cnnmodel.predict_calss(X=[word_test[idx], e1dist_test[idx], e2dist_test[idx], e1type_test[idx], e2type_test[idx]])
# print(list(zip(y_test[idx], y)))
# idx = y_test == 13
# y = cnnmodel.predict_calss(X=[word_test[idx], e1dist_test[idx], e2dist_test[idx], e1type_test[idx], e2type_test[idx]])
# print(list(zip(y_test[idx], y)))
# idx = y_test == 14
# y = cnnmodel.predict_calss(X=[word_test[idx], e1dist_test[idx], e2dist_test[idx], e1type_test[idx], e2type_test[idx]])
# print(list(zip(y_test[idx], y)))
# idx = y_test == 15
# y = cnnmodel.predict_calss(X=[word_test[idx], e1dist_test[idx], e2dist_test[idx], e1type_test[idx], e2type_test[idx]])
# print(list(zip(y_test[idx], y)))

# idx = y_train == 1
# y = cnnmodel.predict_calss(X=[word_train[idx], e1dist_train[idx], e2dist_train[idx], e1type_train[idx], e2type_train[idx]])
# print(list(zip(y_train[idx], y)))
# idx = y_train == 2
# y = cnnmodel.predict_calss(X=[word_train[idx], e1dist_train[idx], e2dist_train[idx], e1type_train[idx], e2type_train[idx]])
# print(list(zip(y_train[idx], y)))
# idx = y_train == 3
# y = cnnmodel.predict_calss(X=[word_train[idx], e1dist_train[idx], e2dist_train[idx], e1type_train[idx], e2type_train[idx]])
# print(list(zip(y_train[idx], y)))
# idx = y_train == 5
# y = cnnmodel.predict_calss(X=[word_train[idx], e1dist_train[idx], e2dist_train[idx], e1type_train[idx], e2type_train[idx]])
# print(list(zip(y_train[idx], y)))
# idx = y_train == 6
# y = cnnmodel.predict_calss(X=[word_train[idx], e1dist_train[idx], e2dist_train[idx], e1type_train[idx], e2type_train[idx]])
# print(list(zip(y_train[idx], y)))
# idx = y_train == 13
# y = cnnmodel.predict_calss(X=[word_train[idx], e1dist_train[idx], e2dist_train[idx], e1type_train[idx], e2type_train[idx]])
# print(list(zip(y_train[idx], y)))
# idx = y_train == 14
# y = cnnmodel.predict_calss(X=[word_train[idx], e1dist_train[idx], e2dist_train[idx], e1type_train[idx], e2type_train[idx]])
# print(list(zip(y_train[idx], y)))
# idx = y_train == 15
# y = cnnmodel.predict_calss(X=[word_train[idx], e1dist_train[idx], e2dist_train[idx], e1type_train[idx], e2type_train[idx]])
# print(list(zip(y_train[idx], y)))
