from gensim.models import Word2Vec
import numpy as np
import tensorflow as tf
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


model = Word2Vec.load("latin_s100_w30_min5_sg_lemmed.model")
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



glove_file = datapath('/home/dbalck/w266_final_project/glove/glove.6B.100d.txt')
tmp_file = get_tmpfile("/home/dbalck/w266_final_project/word2vec/english_word2vec.txt")

# call glove2word2vec script
# default way (through CLI): python -m gensim.scripts.glove2word2vec --input <glove_file> --output <w2v_file>
glove2word2vec(glove_file, tmp_file)

model = KeyedVectors.load_word2vec_format(tmp_file)
