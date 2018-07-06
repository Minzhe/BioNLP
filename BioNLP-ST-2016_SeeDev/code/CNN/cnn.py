##########################################################################################################
###                                              CNN.py                                                ###
##########################################################################################################

from keras.models import Model, load_model
from keras.layers import Input, Embedding, Dropout, Dense
from keras.layers import concatenate, Conv1D, GlobalMaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint


class cnn(object):
    def __init__(self, embeddings, n_label, sent_length, indist_dim, inpos_dim, outdist_dim=32, outpos_dim=4, n_filter=128, win_size=3, dropout=0.25, learning_rate=1e-4):
        self.embeddings = embeddings
        self.n_label = n_label
        self.sent_length = sent_length
        self.indist_dim = indist_dim
        self.inpos_dim = inpos_dim
        self.outdist_dim = outdist_dim
        self.outpos_dim = outpos_dim
        self.n_filter = n_filter
        self.win_size = win_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = self._model()
    
    def _model(self):
        print('Initilizing CNN model ...', end='', flush=True)
        embeddings = self.embeddings
        sent_length = self.sent_length

        # embedding layers
        words_input = Input(shape=(sent_length,), dtype='int32', name='words_input')
        words = Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], weights=[embeddings], trainable=False) (words_input)

        dist1_input = Input(shape=(sent_length,), dtype='int32', name='dist1_input')
        dist1 = Embedding(input_dim=self.indist_dim, output_dim=self.outdist_dim, trainable=True) (dist1_input)

        dist2_input = Input(shape=(sent_length,), dtype='int32', name='dist2_input')
        dist2 = Embedding(input_dim=self.indist_dim, output_dim=self.outdist_dim, trainable=True) (dist2_input)

        pos1_input = Input(shape=(sent_length,), dtype='int32', name='pos1_input')
        pos1 = Embedding(input_dim=self.inpos_dim, output_dim=self.outpos_dim, trainable=True) (pos1_input)

        pos2_input = Input(shape=(sent_length,), dtype='int32', name='pos2_input')
        pos2 = Embedding(input_dim=self.inpos_dim, output_dim=self.outpos_dim, trainable=True) (pos2_input)

        # convolution layer
        conv = concatenate([words, dist1, dist2, pos1, pos2])
        conv = Conv1D(filters=self.n_filter, kernel_size=self.win_size, padding='same', activation='tanh', strides=1) (conv)

        # max pool and softmax
        output = GlobalMaxPooling1D() (conv)
        output = Dropout(self.dropout) (output)
        output = Dense(self.n_label, activation='softmax') (output)

        # model
        model = Model(inputs=[words_input, dist1_input, dist2_input, pos1_input, pos2_input], outputs=[output])
        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        print('Done\nModel structure summary:', flush=True)
        print(model.summary())

        return model
    
    def train(self, X_train, y_train, save_path, validation_split, batch_size, epochs, verbose=2):
        print('Start training CNN models ... ', end='', flush=True)
        early_stopper = EarlyStopping(patience=10, verbose=1)
        check_pointer = ModelCheckpoint(save_path, verbose=1, save_best_only=True)
        self.model.fit(X_train, y_train, 
                       validation_split=validation_split, 
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
        return y.argmax(axis=-1)



