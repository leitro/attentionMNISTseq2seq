import tensorflow as tf
import os
from tensorflow.contrib import layers
import numpy as np
import time
import mnist


class seqAttn():
    def __init__(self, get_data_func, epochs, output_max_len):

        self.tokens = {'GO_TOKEN': 0, 'END_TOKEN': 1, 'UNK_TOKEN': 2}
        self.num_tokens = len(self.tokens.keys())
        self.embed_dim = 4
        self.batch_size = 32
        self.num_units = 256
        self.vocab_size = 10 + self.num_tokens
        self.output_max_len = output_max_len
        self.epochs = epochs
        self.trainImg, trainLabel, self.testImg, testLabel = get_data_func(output_max_len, output_max_len)
        self.trainLabel = np.array(trainLabel) + self.num_tokens
        self.testLabel = np.array(testLabel) + self.num_tokens
        num_train = len(self.trainLabel)
        num_test = len(self.testLabel)
        self.n_per_epoch = num_train // self.batch_size
        self.n_per_epoch_t = num_test // self.batch_size
        self.sample = self.sampler(num_train)
        self.sample_t = self.sampler_t(num_test)
        self.model()

    def createGT(self, n_iter, data, train=True):
        if not os.path.exists('pred_logs'):
            os.makedirs('pred_logs')
        start_n = n_iter*self.batch_size
        if train:
            file_name = 'pred_logs/train_groundtruth.dat'
        else:
            file_name = 'pred_logs/test_groundtruth.dat'
        with open(file_name, 'a') as f:
            for n, seq in enumerate(data):
                f.write(str(start_n+n)+' ')
                for i in seq:
                    f.write(str(i-self.num_tokens))
                f.write('\n')

    def writePredict(self, epoch, n_iter, pred, train=True, trainpre=False): # batch_size, max_output_len
        if not os.path.exists('pred_logs'):
            os.makedirs('pred_logs')
        start_n = n_iter*self.batch_size
        if train:
            if not trainpre:
                file_prefix = 'pred_logs/train_predict_seq.'
            else:
                file_prefix = 'pred_logs/train2_predict_seq.'
        else:
            file_prefix = 'pred_logs/test_predict_seq.'
        with open(file_prefix+str(epoch)+'.log', 'a') as f:
            for n, seq in enumerate(pred):
                f.write(str(start_n+n)+' ')
                for i in seq:
                    f.write(str(i-self.num_tokens))
                f.write('\n')

    def writeLoss(self, loss_value, train=True):
        if not os.path.exists('pred_logs'):
            os.makedirs('pred_logs')
        if train:
            file_name = 'pred_logs/loss_train.log'
        else:
            file_name = 'pred_logs/loss_test.log'
        with open(file_name, 'a') as f:
            f.write(str(loss_value))
            f.write(' ')

    # seqImg (28, 28*5)
    def reshapeSeq(self, seqImg):
        aa = [seqImg[:, i*28:(i+1)*28] for i in range(5)]
        data = [x.reshape(28*28) for x in aa]
        return data # [<28*28>, . . .] num 5

    # seqImgBatch (batch_size, 28, 28*5)
    def reshapeSeqBatch(self, seqImgBatch):
        new_data = []
        for i in seqImgBatch:
            new_data.append(self.reshapeSeq(i))
        return new_data # (batch_size, 5, 28*28)

    def sampler(self, num):
        batches = num // self.batch_size
        while True:
            for i in range(batches):
                in_data = self.trainImg[i*self.batch_size: (i+1)*self.batch_size]
                in_data = self.reshapeSeqBatch(in_data)
                out_data = self.trainLabel[i*self.batch_size: (i+1)*self.batch_size]
                yield {'input_sa': in_data, 'output_sa': out_data}


    def sampler_t(self, num):
        batches = num // self.batch_size
        while True:
            for i in range(batches):
                in_data = self.testImg[i*self.batch_size: (i+1)*self.batch_size]
                in_data = self.reshapeSeqBatch(in_data)
                out_data = self.testLabel[i*self.batch_size: (i+1)*self.batch_size]
                yield {'input_sa_t': in_data, 'output_sa_t': out_data}


    def model(self):
        self.in_data = tf.placeholder(tf.float32, shape=[self.batch_size, self.output_max_len, 28*28])
        self.out_data = tf.placeholder(tf.int32, shape=[self.batch_size, self.output_max_len])

        embedding_matrix = tf.get_variable(
                            'embedding_matrix',
                            shape=(self.vocab_size, self.embed_dim),
                            dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
        # (batch_size, output_max_len, embed_dim)
        embeded = tf.nn.embedding_lookup(embedding_matrix, self.out_data) #not used in the latest version

        # <ENCODER>
        cell1 = tf.contrib.rnn.GRUCell(num_units=self.num_units)
        # (batch_size, 5, 256)   (batch_size, 256)
        encoder_out, encoder_final_state = tf.nn.dynamic_rnn(cell1, self.in_data, dtype=tf.float32)
        # </ENCODER>

        # <DECODER>
        output_lengths = tf.convert_to_tensor([self.output_max_len]*self.batch_size) # not used in the latest version
        train_helper = tf.contrib.seq2seq.TrainingHelper(embeded, output_lengths) # not used in the latest version
        start_tokens = np.array([self.tokens['GO_TOKEN']]*self.batch_size)
        end_token = self.tokens['END_TOKEN']
        test_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                        embedding_matrix, # TODO
                        start_tokens,
                        end_token)
        cell2 = tf.contrib.rnn.GRUCell(num_units=self.num_units)
        attention = tf.contrib.seq2seq.BahdanauAttention(
                        num_units=self.num_units,
                        memory=encoder_out,
                        memory_sequence_length=None)
        cell2_5 = tf.contrib.seq2seq.AttentionWrapper(
                        cell2,
                        attention,
                        attention_layer_size=6) # TODO
        cell3 = tf.contrib.rnn.OutputProjectionWrapper(cell2_5, self.vocab_size)

        init_state = cell3.zero_state(dtype=tf.float32, batch_size=self.batch_size).clone(cell_state=encoder_final_state)
        decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=cell3,
                    #helper=train_helper,
                    helper=test_helper, # tricky, abnormal way to train
                    initial_state=init_state) # (batch_s, num_units) 32, 256
        # (final_outputs, final_state, final_sequence_lengths)
        # final_outputs -> (rnn_output, sample_id) ((32, 5, 10), (32, 5))
        # final_state -> (32, 256)
        # final_sequence_lengths -> (32)
        self.decoder_out = tf.contrib.seq2seq.dynamic_decode(
                        decoder=decoder,
                        output_time_major=False,
                        impute_finished=True,
                        maximum_iterations=self.output_max_len,
                        swap_memory=True) # fix the OOM error

        init_state_2 = cell3.zero_state(dtype=tf.float32, batch_size=self.batch_size).clone(cell_state=encoder_final_state)
        decoder_t = tf.contrib.seq2seq.BasicDecoder(
                        cell=cell3,
                        helper=test_helper,
                        initial_state=init_state_2)
        self.decoder_out_t = tf.contrib.seq2seq.dynamic_decode(
                        decoder=decoder_t,
                        output_time_major=False,
                        impute_finished=True,
                        maximum_iterations=self.output_max_len,
                        swap_memory=True)
        # </DECODER>

        # <LOSS & OPT>
        weights = tf.ones([self.batch_size, self.output_max_len])
        # sequence_loss(logits, targets, weights, ...)
        # logits -> (batch_size, sequence_length, num_decoder_symbols)
        # targets -> (batch_size, sequence_length)
        # weights -> (batch_size, sequence_length)
        self.loss = tf.contrib.seq2seq.sequence_loss(
                self.decoder_out[0].rnn_output,
                self.out_data,
                weights=weights)

        self.train_op = layers.optimize_loss(
                    self.loss,
                    tf.train.get_global_step(),
                    optimizer='Adam',
                    learning_rate=1e-4,
                    summaries=['loss', 'learning_rate'])
        # </LOSS & OPT>

    def train(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.epochs):
                train_loss = 0
                start = time.time()
                for i in range(self.n_per_epoch):
                    data = self.sample.__next__()
                    train_in, train_out = data['input_sa'], data['output_sa']
                    if epoch == 0:
                        self.createGT(i, train_out, True)
                    batch_loss, _, dec_out, dec_out_p = sess.run([self.loss, self.train_op, self.decoder_out[0].sample_id, self.decoder_out_t[0].sample_id], feed_dict={self.in_data: train_in, self.out_data: train_out})
                    self.writePredict(epoch, i, dec_out, True, False)
                    self.writePredict(epoch, i, dec_out_p, True, True)
                    train_loss += batch_loss
                train_loss /= self.n_per_epoch
                self.writeLoss(train_loss, True)
                print('epoch %d/%d, loss=%.3f, time=%.3f' % (epoch, self.epochs, batch_loss, time.time()-start))

                test_loss = 0
                start_t = time.time()
                for j in range(self.n_per_epoch_t):
                    data_t = self.sample_t.__next__()
                    test_in, test_out = data_t['input_sa_t'], data_t['output_sa_t']

                    if epoch == 0: # create groundtruth
                        self.createGT(j, test_out, False)
                    batch_loss_t, dec_out_t = sess.run([self.loss, self.decoder_out_t[0].sample_id], feed_dict={self.in_data: test_in, self.out_data: test_out})
                    test_loss += batch_loss_t
                    self.writePredict(epoch, j, dec_out_t, False)
                test_loss /= self.n_per_epoch_t
                self.writeLoss(test_loss, False)
                print('##TEST## loss=%.3f, time=%.3f' % (test_loss, time.time()-start_t))
                    #import cv2
                    #firstLabel_greedy_t = str(dec_out_t[0]-3)
                    #firstLabel_true_t = str(np.array(test_out[0])-3)
                    #firstImg = np.hstack([x.reshape(28, 28) for x in test_in[0]])
                    #cv2.imwrite('test_imgs/'+str(i)+'_'+firstLabel_true_t+'_'+firstLabel_greedy_t+'.jpg', firstImg*255)

if __name__ == '__main__':
    model = seqAttn(mnist.get_mnist_data, 200, 5)
    model.train()
