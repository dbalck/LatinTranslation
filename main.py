import tensorflow as tf
import time
import model

def run_epoch(e, encoder, decoder, optimizer, loss):
    start = time.time()
    hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset):
        loss = 0
        with tf.GradientTape() as tape:
            enc_output = enc_hidden = encoder(inp, hidden)
            dec_input = tf.expand_dims([targ_lang.word2idx['<start>']] * BATCH_SIZE, 1)

            for t in range(1, targ.shape[1]):
                pred, dec_hidden, _ decoder(dec_input, dec_hidden, enc_output)
                loss += loss_func(targ[:, t], 1)
                dec_input = tf.expand_dims(targ[:, t], 1)

        total_loss += (loss / int(targ.shape[1]))
        variables = encoder.variables + decoder.variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables), tf.train.get_or_create_global_step())

        if batch % 100 == 0
            print("Epoch {} Batch {} Loss {:.4f}".format(e + 1, batch, loss.numpy() / int(targ.shape[1])))
    print("Epoch {} Loss {:.4f} Time {}".format(e + 1, total_loss / len(input_tensor), time.Time() - start))
    

def loss_func(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)

if __name__ == "__main__":

    encoder = model.LatinEncoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
    decoder = model.LatinDecoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
    optimizer = tf.train.AdamOptimizer()

    for e in range(EPOCHS):
        run_epoch(e, encoder, decoder, optimizer)


