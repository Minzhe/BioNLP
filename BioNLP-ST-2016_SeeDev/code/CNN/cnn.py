##########################################################################################################
###                                              CNN.py                                                ###
##########################################################################################################

from keras.models import Model, load_model
from keras.layers import Input, Embedding, Dropout, Dense
from keras.layers import concatenate, Conv1D, GlobalMaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from utility import hLabelEncoder


###########################       function       #########################
def weighted_crossentropy(y_pred, y_true, e):
    '''
    Add hierarchy label loss to prediction loss.
    '''
    label_loss = K.sparse_categorical_crossentropy(target=y_true, output=y_pred)
    h_pred = get_hierarchy(y_pred)
    h_true = get_hierarchy(y_true)
    hierarchy_loss = K.sparse_categorical_crossentropy(target=h_true, output=h_pred)
    return e * hierarchy_loss + (1-e) * label_loss


def get_hierarchy(y):
    '''
    Get the hierarchy of predict label
    '''
    label_encoder = hLabelEncoder()
    return label_encoder.class2hierarchy(y)

#############################    model    ###############################
class cnn(object):
    def __init__(self, model, embeddings, n_label, sent_length, 
                 indist_dim, intype_dim, outdist_dim=32, outtype_dim=16,
                 n_hierarchy=None, loss_weights=None, n_filter=128, win_size=3, dropout=0.25, learning_rate=1e-4):
        self.embeddings = embeddings
        self.n_label = n_label
        self.sent_length = sent_length
        self.indist_dim = indist_dim
        self.intype_dim = intype_dim
        self.outdist_dim = outdist_dim
        self.outtype_dim = outtype_dim
        self.n_hierarchy = n_hierarchy
        self.loss_weights = loss_weights
        self.n_filter = n_filter
        self.win_size = win_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.modelname = model
        if model == 'base':
            self.model = self._model()
        elif model == 'weighted_loss' and self.n_hierarchy is not None and loss_weights is not None:
            self.model = self._model_weighted_loss()
        else:
            raise ValueError('Unrecognized model.')
    
    def _model(self):
        print('Initilizing CNN model ...', end='', flush=True)
        embeddings = self.embeddings
        sent_length = self.sent_length

        ### embedding layers
        # word embedding
        words_input = Input(shape=(sent_length,), dtype='int32', name='words_input')
        words = Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], weights=[embeddings], trainable=False) (words_input)

        # distance embedding
        dist1_input = Input(shape=(sent_length,), dtype='int32', name='dist1_input')
        dist1 = Embedding(input_dim=self.indist_dim, output_dim=self.outdist_dim, trainable=True) (dist1_input)

        dist2_input = Input(shape=(sent_length,), dtype='int32', name='dist2_input')
        dist2 = Embedding(input_dim=self.indist_dim, output_dim=self.outdist_dim, trainable=True) (dist2_input)

        # type embedding
        type1_input = Input(shape=(sent_length,), dtype='int32', name='type1_input')
        type1 = Embedding(input_dim=self.intype_dim, output_dim=self.outtype_dim, trainable=True) (type1_input)

        type2_input = Input(shape=(sent_length,), dtype='int32', name='type2_input')
        type2 = Embedding(input_dim=self.intype_dim, output_dim=self.outtype_dim, trainable=True) (type2_input)

        ### convolution layer
        conv = concatenate([words, dist1, dist2, type1, type2])
        conv = Conv1D(filters=self.n_filter, kernel_size=self.win_size, padding='same', activation='tanh', strides=1) (conv)

        ### max pool and softmax
        output = GlobalMaxPooling1D() (conv)
        output = Dropout(self.dropout) (output)
        output = Dense(self.n_label, activation='softmax') (output)

        # model
        model = Model(inputs=[words_input, dist1_input, dist2_input, type1_input, type2_input], outputs=[output])
        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        print('Done\nModel structure summary:', flush=True)
        print(model.summary())

        return model
    
    def _model_weighted_loss(self):
        print('Initilizing CNN model ...', end='', flush=True)
        embeddings = self.embeddings
        sent_length = self.sent_length

        ### embedding layers
        # word embedding
        words_input = Input(shape=(sent_length,), dtype='int32', name='words_input')
        words = Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], weights=[embeddings], trainable=False) (words_input)

        # distance embedding
        dist1_input = Input(shape=(sent_length,), dtype='int32', name='dist1_input')
        dist1 = Embedding(input_dim=self.indist_dim, output_dim=self.outdist_dim, trainable=True) (dist1_input)

        dist2_input = Input(shape=(sent_length,), dtype='int32', name='dist2_input')
        dist2 = Embedding(input_dim=self.indist_dim, output_dim=self.outdist_dim, trainable=True) (dist2_input)

        # type embedding
        type1_input = Input(shape=(sent_length,), dtype='int32', name='type1_input')
        type1 = Embedding(input_dim=self.intype_dim, output_dim=self.outtype_dim, trainable=True) (type1_input)

        type2_input = Input(shape=(sent_length,), dtype='int32', name='type2_input')
        type2 = Embedding(input_dim=self.intype_dim, output_dim=self.outtype_dim, trainable=True) (type2_input)

        ### convolution layer
        conv = concatenate([words, dist1, dist2, type1, type2])
        conv = Conv1D(filters=self.n_filter, kernel_size=self.win_size, padding='same', activation='tanh', strides=1) (conv)

        ### max pool and softmax
        pool = GlobalMaxPooling1D() (conv)

        ### output layer
        # output label
        output1 = Dropout(self.dropout) (pool)
        output1 = Dense(self.n_label, activation='softmax') (output1)
        # output hierarchy
        output2 = Dropout(self.dropout) (pool)
        output2 = Dense(self.n_hierarchy, activation='softmax') (output2)

        # model
        model = Model(inputs=[words_input, dist1_input, dist2_input, type1_input, type2_input], outputs=[output1, output2])
        model.compile(loss='sparse_categorical_crossentropy', loss_weights=self.loss_weights, optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        print('Done\nModel structure summary:', flush=True)
        print(model.summary())

        return model
    
    def train(self, X_train, y_train, save_path, validation_split, batch_size, epochs, verbose=2):
        # print('Start training CNN models ... ', end='', flush=True)
        early_stopper = EarlyStopping(patience=10, verbose=1)
        check_pointer = ModelCheckpoint(save_path, verbose=1, save_best_only=True)
        self.model.fit(X_train, y_train, 
                       validation_split=0.2, 
                       batch_size=batch_size, 
                       epochs=epochs, 
                       verbose=verbose, 
                       shuffle=True,
                       callbacks=[early_stopper, check_pointer])
        print('Done')

    def loadModel(self, path):
        print('Loading trained CNN model ... ', end='', flush=True)
        self.model = load_model(path)
        print('Done')
    
    def predict(self, X):
        print('Predicting with CNN ... ', flush=True)
        y = self.model.predict(X, verbose=1)
        return y
    
    def predict_calss(self, X):
        y = self.predict(X)
        if self.modelname == 'base':
            return y.argmax(axis=-1)
        elif self.modelname == 'weighted_loss':
            return y[0].argmax(axis=-1)


# if __name__ == '__main__':
#     y_pred = K.constant([0,0,2,2,4])
#     y_true = K.constant([0,1,2,3,4])
#     print(K.eval(y_pred), K.eval(y_true))
#     print(K.eval(weighted_crossentropy(y_pred=y_pred, y_true=y_true, e=0)))
