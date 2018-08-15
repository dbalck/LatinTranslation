from __future__ import absolute_import, division, print_function

import tensorflow as tf
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

tf.enable_eager_execution()

from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import time

optimizer = tf.train.AdamOptimizer()

def create_dataset():
    pairs = []
    for filename in os.listdir("./aligned"):
        with open("./aligned/" + filename, "r") as aligned:
            for line in aligned:
                latin, english = line.split("$$")
                pairs.append((latin, english))
    return pairs

def max_length(tensor):
    return max(len(t) for t in tensor)

def load_dataset():
    pairs = create_dataset()
    inp_lang = LanguageIndex(lat for lat, eng in pairs)
    targ_lang = LanguageIndex(eng for lat, eng in pairs)
    
    # Vectorize the input and target languages
    
    # Spanish sentences
    input_tensor = [[inp_lang.word2idx[s] for s in lat.split(' ')] for lat, eng in pairs]
    
    # English sentences
    target_tensor = [[targ_lang.word2idx[s] for s in eng.split(' ')] for lat, eng in pairs]
    
    # Calculate max_length of input and output tensor
    # Here, we'll set those to the longest sentence in the dataset
    max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)
    
    # Padding the input and output tensor to the maximum length
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, 
                                                                 maxlen=max_length_inp,
                                                                 padding='post')
    
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, 
                                                                  maxlen=max_length_tar, 
                                                                  padding='post')
    
    return input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_tar

class LanguageIndex():
  def __init__(self, lang):
    self.lang = lang
    self.word2idx = {}
    self.idx2word = {}
    self.vocab = set()
    
    self.create_index()
    
  def create_index(self):
    for phrase in self.lang:
      self.vocab.update(phrase.split(' '))
    
    self.vocab = sorted(self.vocab)
    
    self.word2idx['<pad>'] = 0
    for index, word in enumerate(self.vocab):
      self.word2idx[word] = index + 1
    
    for word, index in self.word2idx.items():
      self.idx2word[index] = word


def latin_embeddings():

    #model = Word2Vec.load("word2vec/latin_s100_w30_min5_sg_lemmed.model")
    model = Word2Vec.load("word2vec/latin_s100_w30_min5_sg_lemmed.model")
    VOCAB_LEN = len(model.wv.vocab)
    EMBED_DIM = 100

    # embeddings have a length of 100
    embedding_matrix = np.zeros([VOCAB_LEN, EMBED_DIM])
    for i in range(VOCAB_LEN):
        embedding_vector = model.wv[model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    #print(embedding_matrix)
    #saved_embeddings = tf.constant(embedding_matrix)
    #embedding = tf.Variable(initial_value=saved_embeddings, trainable=False)

    embeddings = tf.keras.layers.Embedding(input_dim=VOCAB_LEN, output_dim=EMBED_DIM, weights=[embedding_matrix])
    return embeddings

def english_embeddings():
    model = KeyedVectors.load_word2vec_format("word2vec/english_word2vec.txt")
    VOCAB_LEN = len(model.wv.vocab)
    EMBED_DIM = 100

    # embeddings have a length of 100
    embedding_matrix = np.zeros([VOCAB_LEN, EMBED_DIM])
    for i in range(VOCAB_LEN):
        embedding_vector = model.wv[model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    #print(embedding_matrix)
    #saved_embeddings = tf.constant(embedding_matrix)
    #embedding = tf.Variable(initial_value=saved_embeddings, trainable=False)

    embeddings = tf.keras.layers.Embedding(input_dim=VOCAB_LEN, output_dim=EMBED_DIM, weights=[embedding_matrix])
    return embeddings

def gru(units):
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNGRU(
                units,
                return_sequences=True,
                return_state=True,
                recurrent_initializer="glorot_uniform")
    else:
        return tf.keras.layers.GRU(
                units,
                return_sequences=True,
                return_state=True,
                recurrent_activation="sigmoid",
                recurrent_initializer="glorot_uniform")

class LatinEncoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(LatinEncoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim) 
        #self.embedding = self.latin_embeddings()  
        self.gru = gru(self.enc_units)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units))

    def latin_embeddings(self):
        model = Word2Vec.load("word2vec/latin_s100_w30_min5_sg_lemmed.model")
        VOCAB_LEN = len(model.wv.vocab)
        EMBED_DIM = 100

        # embeddings have a length of 100
        embedding_matrix = np.zeros([VOCAB_LEN, EMBED_DIM])
        for i in range(VOCAB_LEN):
            embedding_vector = model.wv[model.wv.index2word[i]]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        embeddings = tf.keras.layers.Embedding(input_dim=VOCAB_LEN, output_dim=EMBED_DIM, weights=[embedding_matrix])
        return embeddings


class LatinDecoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
        super(LatinDecoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.gru = gru(self.dec_units)
        #self.embedding = self.english_embeddings() 
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim) 
        self.fc = tf.keras.layers.Dense(vocab_size)

        # attention params
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, x, hidden, enc_output):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)

        return x, state, attention_weights

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.dec_units))

    def english_embeddings(self):
        model = KeyedVectors.load_word2vec_format("word2vec/english_word2vec.txt")
        VOCAB_LEN = len(model.wv.vocab)
        EMBED_DIM = 100
        # embeddings have a length of 100
        embedding_matrix = np.zeros([VOCAB_LEN, EMBED_DIM])
        for i in range(VOCAB_LEN):
            embedding_vector = model.wv[model.wv.index2word[i]]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        #print(embedding_matrix)
        #saved_embeddings = tf.constant(embedding_matrix)
        #embedding = tf.Variable(initial_value=saved_embeddings, trainable=False)

        embeddings = tf.keras.layers.Embedding(input_dim=VOCAB_LEN, output_dim=EMBED_DIM, weights=[embedding_matrix])
        return embeddings

def translate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    result, sentence, attention_plot = evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
        
    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(result))
    
    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))

def loss_function(real, pred):
  mask = 1 - np.equal(real, 0)
  loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
  return tf.reduce_mean(loss_)
        
if __name__ == "__main__":
    print("start")

    input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_targ = load_dataset()
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

    print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))


    BUFFER_SIZE = len(input_tensor_train)
    BATCH_SIZE = 16 
    N_BATCH = BUFFER_SIZE//BATCH_SIZE
    embedding_dim = 100 
    units = 256
    vocab_inp_size = len(inp_lang.word2idx)
    vocab_tar_size = len(targ_lang.word2idx)
    print("vocab sizes:")
    print(vocab_inp_size)
    print(vocab_tar_size)

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(BATCH_SIZE))

    encoder = LatinEncoder(vocab_inp_size, 100, units, BATCH_SIZE)
    decoder = LatinDecoder(vocab_tar_size, 100, units, BATCH_SIZE)

    EPOCHS = 6

    for epoch in range(EPOCHS):
        start = time.time()
        
        hidden = encoder.initialize_hidden_state()
        total_loss = 0
        
        for (batch, (inp, targ)) in enumerate(dataset):
            loss = 0
            
            with tf.GradientTape() as tape:
                enc_output, enc_hidden = encoder(inp, hidden)
                
                dec_hidden = enc_hidden
                
                dec_input = tf.expand_dims([targ_lang.word2idx['<start>']] * BATCH_SIZE, 1)       
                
                # Teacher forcing - feeding the target as the next input
                for t in range(1, targ.shape[1]):
                    # passing enc_output to the decoder
                    predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                    
                    loss += loss_function(targ[:, t], predictions)
                    
                    # using teacher forcing
                    dec_input = tf.expand_dims(targ[:, t], 1)
            
            batch_loss = (loss / int(targ.shape[1]))
            
            total_loss += batch_loss
            
            variables = encoder.variables + decoder.variables
            
            gradients = tape.gradient(loss, variables)
            
            optimizer.apply_gradients(zip(gradients, variables), tf.train.get_or_create_global_step())
            
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
        
        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / N_BATCH))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    translate('Urbem Romam a principio reges habuere; libertatem et consulatum Lucius instituit.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
    translate('Hoc eo consilio fecit nequis sibi morae quicquam fore speraret et ut omnes in dies horasque parati essent. Incidit per id tempus ut tempestates ad navigandum idoneas non haberet.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
    translate('Caesar docuit', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
    translate('Pedites interim resistebant, dum equites rursus cursu renovato peditibus suis succurrerent. ', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
    translate('Quas singulas scaphae adversariorum complures adortae incendebant atque expugnabant.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)

    


