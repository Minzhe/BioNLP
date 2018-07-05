##########################################################################################################
###                                              CNN.py                                                ###
##########################################################################################################

from keras.models import Model
from keras.layers import Input, Embedding, Dropout, Dense
from keras.layers import concatenate, Convolution1D, GlobalMaxPooling1D
from keras.optimizers import Adam


class cnn(object):
    def __init__(self, embeddings, n_label, sent_length, max_dist, dist_dims, n_filter, filter_length, learning_rate):
        self.embeddings = embeddings
        self.n_label = n_label
        self.sent_length = sent_length
        self.max_dist = max_dist
        self.dist_dims = dist_dims
        self.n_filter = n_filter
        self.filter_length = filter_length
        self.learning_rate = learning_rate
        self.model = self._model()
    
    def _model(self):
        print('Initilizing CNN model ...', end='', flush=True)
        embeddings = self.embeddings
        sent_length = self.sent_length
        dist_dims = self.dist_dims
        max_dist = self.max_dist

        # embedding layers
        words_input = Input(shape=(sent_length,), dtype='int32', name='words_input')
        words = Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], weights=[embeddings], trainable=False) (words_input)

        dist1_input = Input(shape=(sent_length,), dtype='int32', name='dist1_input')
        dist1 = Embedding(input_dim=max_dist, output_dim=dist_dims, trainable=True) (dist1_input)

        dist2_input = Input(shape=(sent_length,), dtype='int32', name='dist2_input')
        dist2 = Embedding(input_dim=max_dist, output_dim=dist_dims, trainable=True) (dist2_input)

        loc1_input = Input(shape=(sent_length,), dtype='int32', name='loc1_input')
        loc1 = Embedding(input_dim=max_dist, output_dim=dist_dims, trainable=True) (loc1_input)

        loc2_input = Input(shape=(sent_length,), dtype='int32', name='loc1_input')
        loc2 = Embedding(input_dim=max_dist, output_dim=dist_dims, trainable=True) (loc2_input)

        # convolution layer
        conv = concatenate([words, dist1, dist2, loc1, loc2])
        conv = Convolution1D(filters=self.n_filter, kernel_size=self.filter_length, padding='same', activation='tanh', strides=1) (conv)

        # max pool and softmax
        output = GlobalMaxPooling1D() (conv)
        output = Dropout(0.25) (output)
        output = Dense(self.n_label, activation='softmax') (output)

        # model
        model = Model(inputs=[words_input, dist1_input, dist2_input, loc1_input, loc2_input], outputs=[output])
        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        print('Done\nModel structure summary:', flush=True)
        print(model.summary())

        return model



