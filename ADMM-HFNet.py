import tensorflow as tf
import numpy as np
from functions import checkFile
import scipy.io as sio
import time
from skimage.measure import compare_psnr
import matplotlib.pyplot as plt
from quality_measure import SAM
from quality_measure import quality_reference_accessment
import tensorflow.contrib.layers as ly
import math
from functions import generateRandomList
import cv2


class deepNet(object):
    '''
    the implementation of ADMM-HFNet
    Shen D, Liu J, Wu Z, et al. Admm-hfnet: A matrix decomposition based deep approach for hyperspectral image fusion[J].
    IEEE Transactions on Geoscience and Remote Sensing, 2021, early access, doi:10.1109/TGRS.2021.3112181.
    '''

    def __init__(self, num=0):

        self.train_batch_size = 16  # 64
        self.valid_batch_size = 16  # 16
        # 32
        self.data_num = num

        self.weight_decay = 2e-5

        self.choose_dataset(num)

        self.ratio = self.test_height / self.test_lheight  # ratio
        self.r1 = math.floor(math.sqrt(self.ratio))  # r1
        self.r2 = self.ratio / self.r1

        self.data = tf.placeholder(dtype=tf.float32, shape=[None, None, None, self.channels], name='Y')
        self.ms_data = tf.placeholder(dtype=tf.float32, shape=[None, None, None, self.mschannels], name='Z')
        self.label = tf.placeholder(dtype=tf.float32, shape=[None, None, None, self.channels], name='X')
        self.P = tf.placeholder(dtype=tf.float32, shape=[None, self.channels, None], name='P')
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')

        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')

        self.helplist = []
        self.losslist1 = []
        self.losslist2 = []

        checkFile(self.model_save_path)

    def choose_dataset(self, num):
        # CAVE
        if num == 0:
            self.channels = 31
            self.mschannels = 3
            self.train_rank = 20
            self.rank = 8
            self.train_pieces_path = r'CAVEMAT/pieces/normal/train/'
            self.valid_pieces_path = r'CAVEMAT/pieces/normal/valid/'
            self.test_data_label_path = r'CAVEMAT/'
            self.output_save_path = r'CAVEMAT/outputs/'
            self.train_pieces_path = r'CAVEMAT/pieces/normal/train/'
            self.valid_pieces_path = r'CAVEMAT/pieces/normal/valid/'
            self.model_save_path = r'models/CAVEMAT/'
            self.output_save_path = r'CAVEMAT/outputs/'

            self.maxpower = 20
            self.total_num = 13456
            self.valid_num = 3364
            self.test_start = 21
            self.test_end = 32

            self.test_height = 512
            self.test_width = 512
            self.test_lheight = 64
            self.test_lwidth = 64

            self.piece_size = 64
            self.lpiece_size = 8
            self.stages = 11

            # HARVARD
        elif num == 1:
            self.channels = 31
            self.mschannels = 3
            self.train_rank = 20
            self.rank = 8

            self.train_pieces_path = r'HARVARDMAT/pieces/normal/train/'
            self.valid_pieces_path = r'HARVARDMAT/pieces/normal/valid/'
            self.test_data_label_path = r'HARVARDMAT/'
            self.output_save_path = r'HARVARDMAT/outputs/'
            self.train_pieces_path = r'HARVARDMAT/pieces/normal/train/'
            self.valid_pieces_path = r'HARVARDMAT/pieces/normal/valid/'
            self.model_save_path = r'models/HARVARDMAT/'

            self.total_num = 124992
            self.valid_num = 31248

            self.maxpower = 20
            self.test_start = 31
            self.test_end = 50
            self.test_height = 1040
            self.test_width = 1392
            self.test_lheight = 130
            self.test_lwidth = 174

            self.piece_size = 64
            self.lpiece_size = 8
            self.stages = 10

        elif num == 2:  # UH ratio = 20
            self.channels = 48
            self.mschannels = 3
            self.train_rank = 4
            self.rank = 4
            self.test_data_label_path = r'HOUSTANMAT/'
            self.output_save_path = r'HOUSTANMAT/outputs/'
            self.train_pieces_path = r'HOUSTANMAT/pieces/normal/train/'
            self.valid_pieces_path = r'HOUSTANMAT/pieces/normal/valid/'
            self.model_save_path = r'models/HOUSTANMAT/'

            self.maxpower = 20
            self.total_num = 16500
            self.valid_num = 4125
            self.test_start = 1
            self.test_end = 5
            self.test_height = 800
            self.test_width = 600
            self.test_lheight = 40
            self.test_lwidth = 30

            self.piece_size = 60
            self.lpiece_size = 3
            self.stages = 11

        elif num == 3:
            self.channels = 31
            self.mschannels = 3
            self.train_rank = 20
            self.rank = 8
            self.train_pieces_path = r'CAVEMAT2/pieces/normal/train/'
            self.valid_pieces_path = r'CAVEMAT2/pieces/normal/valid/'
            self.test_data_label_path = r'CAVEMAT2/'
            self.output_save_path = r'CAVEMAT2/outputs/'
            self.train_pieces_path = r'CAVEMAT2/pieces/normal/train/'
            self.valid_pieces_path = r'CAVEMAT2/pieces/normal/valid/'
            self.model_save_path = r'models/CAVEMAT2/'

            self.maxpower = 20
            self.total_num = 13456
            self.valid_num = 3364
            self.test_start = 21
            self.test_end = 32

            self.test_height = 512
            self.test_width = 512
            self.test_lheight = 93
            self.test_lwidth = 93

            self.piece_size = 64
            self.lpiece_size = 12
            self.stages = 11

    def soft(self, X, sigma):
        t1 = tf.sign(X)
        t2 = tf.abs(X) - sigma
        t3 = tf.zeros_like(X)
        t4 = tf.where(t2 < 0, x=t3, y=t2)
        return t1 * t4

    def spectralDegrading(self, X, stage, num_ms_spectral=3, weight_decay=2e-5):
        with tf.variable_scope('spectral_down_stage%d' % stage):
            Z_es = ly.conv2d(X, num_outputs=num_ms_spectral, kernel_size=3, stride=1,
                             weights_regularizer=ly.l2_regularizer(weight_decay),
                             weights_initializer=ly.variance_scaling_initializer(), activation_fn=tf.nn.leaky_relu,
                             reuse=tf.AUTO_REUSE, scope='z_es')
            return Z_es

    def spectralUpsampling(self, X, stage, num_spectral=31, weight_decay=2e-5):
        with tf.variable_scope('spectral_up_stage%d' % stage):
            Z_back = ly.conv2d(X, num_outputs=num_spectral, kernel_size=3, stride=1,
                               weights_regularizer=ly.l2_regularizer(weight_decay),
                               weights_initializer=ly.variance_scaling_initializer(), activation_fn=None,
                               reuse=tf.AUTO_REUSE, scope='EZ')
            return Z_back

    def BDOperator(self, X, stage, H=512, W=512, h=64, w=64, num_spectral=31, weight_decay=2e-5):
        with tf.variable_scope('BD_stage%d' % stage):
            r1 = self.r1
            r2 = self.r2

            if int(r2) == r2:
                output1 = ly.conv2d(X, num_outputs=num_spectral, kernel_size=int(r2 + 4), stride=int(r2),
                                    weights_regularizer=ly.l2_regularizer(weight_decay),
                                    weights_initializer=ly.variance_scaling_initializer(),
                                    activation_fn=tf.nn.leaky_relu,
                                    reuse=tf.AUTO_REUSE, scope='output1')
                output2 = output1
                output3 = ly.conv2d(output2, num_outputs=num_spectral, kernel_size=r1 + 4, stride=r1,
                                    weights_regularizer=ly.l2_regularizer(weight_decay),
                                    weights_initializer=ly.variance_scaling_initializer(),
                                    activation_fn=tf.nn.leaky_relu,
                                    reuse=tf.AUTO_REUSE, scope='output3')

            else:

                output1 = tf.image.resize_bilinear(X, (r1 * h, r1 * w), name='blinear')
                output1 = ly.conv2d(output1, num_outputs=num_spectral, kernel_size=5, stride=1,
                                    weights_regularizer=ly.l2_regularizer(weight_decay),
                                    weights_initializer=ly.variance_scaling_initializer(),
                                    activation_fn=tf.nn.leaky_relu,
                                    reuse=tf.AUTO_REUSE, scope='output1')
                output2 = output1
                output3 = ly.conv2d(output2, num_outputs=num_spectral, kernel_size=r1 + 4, stride=r1,
                                    weights_regularizer=ly.l2_regularizer(weight_decay),
                                    weights_initializer=ly.variance_scaling_initializer(),
                                    activation_fn=tf.nn.leaky_relu,
                                    reuse=tf.AUTO_REUSE, scope='output3')

            return output1, output2, output3

    def UpResAdjust(self, name, X, Y, num_spectral, ms_num_spectral, layer_num=3, weight_decay=2e-5):
        with tf.variable_scope(name):
            output = tf.concat([X, Y], axis=-1)
            for i in range(1, layer_num):
                output = ly.conv2d(output, num_outputs=num_spectral + ms_num_spectral, kernel_size=3, stride=1,
                                   weights_regularizer=ly.l2_regularizer(weight_decay),
                                   weights_initializer=ly.variance_scaling_initializer(),
                                   activation_fn=tf.nn.leaky_relu,
                                   reuse=tf.AUTO_REUSE, scope='output%d' % i)
            output = ly.conv2d(output, num_outputs=num_spectral, kernel_size=3, stride=1,
                               weights_regularizer=ly.l2_regularizer(weight_decay),
                               weights_initializer=ly.variance_scaling_initializer(), activation_fn=tf.nn.leaky_relu,
                               reuse=tf.AUTO_REUSE, scope='output%d' % layer_num)
            return X + output

    def BDTOperator(self, X, stage, Z, Z1, H=512, W=512, h=64, w=64, num_spectral=31, ms_spectral=3,
                    weight_decay=2e-5):
        with tf.variable_scope('BDT_stage%d' % stage):
            r1 = self.r1
            r2 = self.r2
            if int(r2) == r2:
                output1 = ly.conv2d_transpose(X, num_spectral, r1 + 4, r1, activation_fn=None,
                                              weights_initializer=ly.variance_scaling_initializer(),
                                              weights_regularizer=ly.l2_regularizer(weight_decay), reuse=tf.AUTO_REUSE,
                                              scope="output1")
                output1 = self.UpResAdjust('prior1', output1, Z1, num_spectral=num_spectral,
                                           ms_num_spectral=ms_spectral)
                output2 = output1
                output3 = ly.conv2d_transpose(output2, num_spectral, int(r2 + 4), int(r2), activation_fn=None,
                                              weights_initializer=ly.variance_scaling_initializer(),
                                              weights_regularizer=ly.l2_regularizer(weight_decay), reuse=tf.AUTO_REUSE,
                                              scope="output3")
                output3 = self.UpResAdjust('prior3', output3, Z, num_spectral=num_spectral, ms_num_spectral=ms_spectral)
            else:
                output1 = ly.conv2d_transpose(X, num_spectral, r1 + 4, r1, activation_fn=None,
                                              weights_initializer=ly.variance_scaling_initializer(),
                                              weights_regularizer=ly.l2_regularizer(weight_decay), reuse=tf.AUTO_REUSE,
                                              scope="output1")
                # print(output1.get_shape())
                output1 = self.UpResAdjust('prior1', output1, Z1, num_spectral=num_spectral,
                                           ms_num_spectral=ms_spectral)
                output2 = output1
                output3 = tf.image.resize_bilinear(output2, (H, W), name='binear')
                output3 = ly.conv2d_transpose(output3, num_spectral, 5, 1, activation_fn=None,
                                              weights_initializer=ly.variance_scaling_initializer(),
                                              weights_regularizer=ly.l2_regularizer(weight_decay), reuse=tf.AUTO_REUSE,
                                              scope="output3")
                output3 = self.UpResAdjust('prior3', output3, Z, num_spectral=num_spectral, ms_num_spectral=ms_spectral)
            return output3

    def buildGraph(self, OH, OW, lh, lw, rank, iterations):
        '''
        network
        :return:
        '''
        with tf.variable_scope('fusion_net'):
            # optimize the efficient matrix at first

            # the first stage (initializing)
            P = self.P
            Z = self.ms_data
            Y = self.data
            PT = tf.transpose(P, (0, 2, 1))

            V = tf.zeros([tf.shape(P)[0], rank, OH * OW])
            A = tf.zeros_like(V)
            G = tf.zeros_like(V)
            T = self.spectralUpsampling(Z, 0, num_spectral=self.channels)
            H = tf.zeros_like(T)

            Z1, Z2, _ = self.BDOperator(Z, 0, OH, OW, lh, lw, num_spectral=self.channels)
            PV = self.BDTOperator(Y, 0, Z, Z1, OH, OW, lh, lw, num_spectral=self.channels,
                                  ms_spectral=self.mschannels)

            # Keep the outputs of all stages
            Xs = []

            # 2-K stages
            for i in range(1, iterations + 1):
                # the updating of V
                D1 = PV
                RPV = self.spectralDegrading(D1, i, num_ms_spectral=self.mschannels)  # RPV
                D2 = RPV - Z

                D3 = self.spectralUpsampling(D2, i, num_spectral=self.channels)
                D3 = tf.reshape(D3, [-1, OH * OW, self.channels])
                D3 = tf.transpose(D3, (0, 2, 1))
                D4 = tf.matmul(PT, D3)

                a5 = tf.Variable(tf.truncated_normal([], stddev=0.1, dtype=tf.float32), name='stage%d_t5' % i)

                D5 = T - a5 * H
                D5 = tf.reshape(D5, [-1, OH * OW, self.channels])
                D5 = tf.transpose(D5, (0, 2, 1))

                D5 = tf.matmul(PT, D5)

                a0 = tf.Variable(tf.truncated_normal([], stddev=0.1, dtype=tf.float32), name='stage%d_t0' % i)
                a1 = tf.Variable(tf.truncated_normal([], stddev=0.1, dtype=tf.float32), name='stage%d_t1' % i)
                a2 = tf.Variable(tf.truncated_normal([], stddev=0.1, dtype=tf.float32), name='stage%d_t2' % i)
                a3 = tf.Variable(tf.truncated_normal([], stddev=0.1, dtype=tf.float32), name='stage%d_t3' % i)
                a4 = tf.Variable(tf.truncated_normal([], stddev=0.1, dtype=tf.float32), name='stage%d_t4' % i)

                V = a0 * V - a1 * D4 + a2 * A + a3 * G + a4 * D5

                PV = tf.matmul(P, V)
                PVT = tf.transpose(PV, (0, 2, 1))
                PV = tf.reshape(PVT, [-1, OH, OW, self.channels])

                # the updating of T
                _, _, TBD = self.BDOperator(T, i, OH, OW, lh, lw, num_spectral=self.channels)
                E1 = TBD - Y
                E2 = self.BDTOperator(E1, i, Z, Z1, OH, OW, lh, lw, num_spectral=self.channels,
                                      ms_spectral=self.mschannels)
                E3 = PV

                b0 = tf.Variable(tf.truncated_normal([], stddev=0.1, dtype=tf.float32), name='stage%d_h0' % i)
                b1 = tf.Variable(tf.truncated_normal([], stddev=0.1, dtype=tf.float32), name='stage%d_h1' % i)
                b2 = tf.Variable(tf.truncated_normal([], stddev=0.1, dtype=tf.float32), name='stage%d_h2' % i)
                b3 = tf.Variable(tf.truncated_normal([], stddev=0.1, dtype=tf.float32), name='stage%d_h3' % i)

                T = b0 * T - b1 * E2 + b2 * E3 + b3 * H

                # the updating of A
                c0 = tf.Variable(tf.truncated_normal([], stddev=0.1, dtype=tf.float32), name='stage%d_g0' % i)
                c1 = tf.Variable(tf.truncated_normal([], stddev=0.1, dtype=tf.float32), name='stage%d_g1' % i)
                A = self.soft(V - c0 * G, c1)

                PA = tf.matmul(P, A)
                PAT = tf.transpose(PA, (0, 2, 1))
                PA = tf.reshape(PAT, [-1, OH, OW, self.channels])

                # the updating of G and H
                eta1 = tf.Variable(tf.truncated_normal([], stddev=0.1, dtype=tf.float32), name='stage%d_eta1' % i)
                eta2 = tf.Variable(tf.truncated_normal([], stddev=0.1, dtype=tf.float32), name='stage%d_eta2' % i)
                eta3 = tf.Variable(tf.truncated_normal([], stddev=0.1, dtype=tf.float32), name='stage%d_eta3' % i)

                mu1 = tf.Variable(tf.truncated_normal([], stddev=0.1, dtype=tf.float32), name='stage%d_mu1' % i)
                mu2 = tf.Variable(tf.truncated_normal([], stddev=0.1, dtype=tf.float32), name='stage%d_mu2' % i)
                mu3 = tf.Variable(tf.truncated_normal([], stddev=0.1, dtype=tf.float32), name='stage%d_mu3' % i)

                G = eta1 * G + eta2 * A - eta3 * V
                H = mu1 * H + mu2 * PV - mu3 * T

                # the updating of X (information supplement model)
                _, _, TBD = self.BDOperator(T, i, OH, OW, lh, lw, num_spectral=self.channels)

                J3 = TBD - Y
                # print(Y.get_shape())
                J3 = self.BDTOperator(J3, i, Z, Z1, OH, OW, lh, lw, num_spectral=self.channels,
                                      ms_spectral=self.mschannels)
                J4 = self.spectralDegrading(T, i, num_ms_spectral=self.mschannels) - Z
                J4 = self.spectralUpsampling(J4, i, num_spectral=self.channels)

                J1 = T - PA
                J2 = T - PV

                e1 = tf.Variable(tf.truncated_normal([], stddev=0.1, dtype=tf.float32), name='stage%d_d1' % i)
                e2 = tf.Variable(tf.truncated_normal([], stddev=0.1, dtype=tf.float32), name='stage%d_d2' % i)
                e3 = tf.Variable(tf.truncated_normal([], stddev=0.1, dtype=tf.float32), name='stage%d_d3' % i)
                e4 = tf.Variable(tf.truncated_normal([], stddev=0.1, dtype=tf.float32), name='stage%d_d4' % i)

                X = T - e1 * J4 - e2 * J3 - e3 * J1 - e4 * J2

                Xs.append(X)

            output = tf.concat(Xs, axis=-1)
            output = ly.conv2d(output, num_outputs=self.channels, kernel_size=3, stride=1,
                               weights_regularizer=ly.l2_regularizer(self.weight_decay),
                               weights_initializer=ly.variance_scaling_initializer(), activation_fn=None, scope='final')

            self.output = output
            self.lastX = X

    def initGpu(self):
        '''
        the setting of gpu
        :return:
        '''
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

    def init_train_steps(self):
        self.initial_lr = 2e-3
        self.end_lr = 1e-7
        self.perdecay_epoch = 10
        self.epoch_completed = 0
        self.currnt_epoch = 0
        self.clipping_norm = 5
        self.l2_decay = 1e-4

        self.lr = self.initial_lr

        self.train_batch_input = np.zeros([self.train_batch_size, self.lpiece_size, self.lpiece_size, self.channels])
        self.train_batch_ms = np.zeros([self.train_batch_size, self.piece_size, self.piece_size, self.mschannels])
        self.train_batch_label = np.zeros([self.train_batch_size, self.piece_size, self.piece_size, self.channels])
        self.train_batch_p = np.zeros([self.train_batch_size, self.channels, self.train_rank])

        self.valid_batch_input = np.zeros([self.valid_batch_size, self.lpiece_size, self.lpiece_size, self.channels])
        self.valid_batch_ms = np.zeros([self.valid_batch_size, self.piece_size, self.piece_size, self.mschannels])
        self.valid_batch_label = np.zeros([self.valid_batch_size, self.piece_size, self.piece_size, self.channels])
        self.valid_batch_p = np.zeros([self.valid_batch_size, self.channels, self.train_rank])
        self.train_step = 0

    def buildOptimizaer(self):
        self.buildPSNRAndSSIM()
        self.buildLoss()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.optimizer = optimizer.minimize(self.loss)

    def buildSaver(self):
        '''
        saver for saving model
        :return:
        '''
        self.saver = tf.train.Saver(max_to_keep=5)
        checkFile(self.model_save_path)

    def buildPSNRAndSSIM(self):
        self.psnr = tf.reduce_mean(tf.image.psnr(self.label, self.output, max_val=1.0))
        self.ssim = tf.reduce_mean(tf.image.ssim(self.label, self.output, max_val=1.0))

    def loadTrainBatch(self):
        self.helplist.clear()
        generateRandomList(self.helplist, self.total_num, self.train_batch_size)
        for ind, val in enumerate(self.helplist):
            mat = sio.loadmat(self.train_pieces_path + '%d.mat' % val)
            # print(val)
            self.train_batch_input[ind, :, :, :] = mat['Y']
            self.train_batch_ms[ind, :, :, :] = mat['Z']
            self.train_batch_label[ind, :, :, :] = mat['label']
            self.train_batch_p[ind, :, :] = mat['P']

    def loadValidBatch(self):
        self.helplist.clear()
        generateRandomList(self.helplist, self.valid_num, self.valid_batch_size)
        for ind, val in enumerate(self.helplist):
            mat = sio.loadmat(self.valid_pieces_path + '%d.mat' % val)
            self.valid_batch_input[ind, :, :, :] = mat['Y']
            self.valid_batch_ms[ind, :, :, :] = mat['Z']
            self.valid_batch_label[ind, :, :, :] = mat['label']
            self.valid_batch_p[ind, :, :] = mat['P']

    def buildLoss(self):
        # L1-norm as the spatial loss
        self.loss = tf.reduce_mean(tf.abs(self.output - self.label), name='spt_loss')

    def epochManipulate(self):
        '''
        every 10 epoch to decay 0.5 until less than 1e-7
        :return:
        '''
        self.epoch_completed += 1
        if self.epoch_completed >= self.perdecay_epoch:
            self.epoch_completed = 0
            self.lr = 0.5 * self.lr

    def trainBatch(self, sess):
        self.loadTrainBatch()
        sess.run(self.optimizer, feed_dict={self.data: self.train_batch_input, self.label: self.train_batch_label,
                                            self.ms_data: self.train_batch_ms,
                                            self.learning_rate: self.lr,
                                            self.P: self.train_batch_p})

    def paintTrend(self):
        plt.title('loss-trend')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.plot(self.losslist1, color='r')
        plt.plot(self.losslist2, color='g')
        plt.legend(['train', 'valid'])
        plt.show()

    def endAllEpochs(self):
        self.helplist.clear()
        print('training is finished')
        self.paintTrend()
        self.losslist1.clear()
        self.losslist2.clear()

    def printStatus(self, sess, iterations):
        psnr_val, ssim_val, loss_val, lr = self.session.run([self.psnr, self.ssim, self.loss, self.learning_rate],
                                                            feed_dict={self.data: self.train_batch_input,
                                                                       self.label: self.train_batch_label,
                                                                       self.ms_data: self.train_batch_ms,
                                                                       self.learning_rate: self.lr,
                                                                       self.P: self.train_batch_p})
        # self.losslist1.append(loss_val)
        print('Iteration%s----train-----psnr:%s  ssim:%s  loss:%s  lr:%s-----' % (
            iterations, psnr_val, ssim_val, loss_val, lr))

    def selectModel(self, sess, iterations):
        self.loadValidBatch()
        psnr_val, ssim_val, loss_val, lr = self.session.run([self.psnr, self.ssim, self.loss, self.learning_rate],
                                                            feed_dict={self.data: self.train_batch_input,
                                                                       self.label: self.train_batch_label,
                                                                       self.ms_data: self.train_batch_ms,
                                                                       self.learning_rate: self.lr,
                                                                       self.P: self.train_batch_p})
        psnr_val2, ssim_val2, loss_val2 = self.session.run([self.psnr, self.ssim, self.loss],
                                                           feed_dict={self.data: self.valid_batch_input,
                                                                      self.label: self.valid_batch_label,
                                                                      self.ms_data: self.valid_batch_ms,
                                                                      self.learning_rate: self.lr,
                                                                      self.P: self.valid_batch_p})
        self.losslist1.append(loss_val)
        self.losslist2.append(loss_val2)

        print('     ----valid-----psnr:%s  ssim:%s  loss:%s-----' % (psnr_val2, ssim_val2, loss_val2))

        t = psnr_val2 * 0.5 + ssim_val2 * 0.5
        if t > self.maxpower:
            print('get a satisfying model')
            self.maxpower = t
            self.saver.save(sess, self.model_save_path, global_step=iterations)
            # print('one model saved successfully')

    def train(self):
        self.initGpu()
        self.init_train_steps()
        self.buildGraph(self.piece_size, self.piece_size, self.lpiece_size, self.lpiece_size, self.train_rank,
                        self.stages)
        self.buildOptimizaer()
        self.buildSaver()
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(1, 160001):
                if i >= 0 and i <= 20000:
                    LR = 4e-4
                elif i > 20000 and i <= 60000:
                    LR = 2e-4
                elif i > 60000 and i <= 1400000:
                    LR = 1e-4
                else:
                    LR = 5e-5
                self.lr = LR
                self.trainBatch(sess)
                if i % 100 == 0:
                    self.printStatus(sess, i)
                if i % 10000 == 0:
                    self.selectModel(sess, i)
            self.endAllEpochs()

    def retrain(self):
        '''
        Breakpoint training
        :return:
        '''
        self.initGpu()
        self.init_train_steps()
        self.buildGraph(self.piece_size, self.piece_size, self.lpiece_size, self.lpiece_size, self.train_rank,
                        self.stages)
        self.buildOptimizaer()
        self.buildSaver()
        with self.session as sess:
            # sess.run(tf.global_variables_initializer())
            latest_model = tf.train.get_checkpoint_state(self.model_save_path)
            self.saver.restore(sess, latest_model.model_checkpoint_path)
            str = latest_model.model_checkpoint_path.split('/')[-1]
            start = int(str.split('-')[1])
            for i in range(start + 1, 160001):
                if i >= 0 and i <= 20000:
                    LR = 4e-4
                elif i > 20000 and i <= 60000:
                    LR = 2e-4
                elif i > 60000 and i <= 1400000:
                    LR = 1e-4
                else:
                    LR = 5e-5
                self.lr = LR
                self.trainBatch(sess)
                if i % 100 == 0:
                    self.printStatus(sess, i)
                if i % 10000 == 0:
                    self.selectModel(sess, i)

            self.endAllEpochs()

    def test(self):
        start = time.perf_counter()
        self.initGpu()
        # the orignal image is too large to be taken into the model directly, we divide it and then group
        if self.data_num == 0:
            # CAVE
            piece_size_x = 512
            piece_size_y = 512
            lpiece_size_x = 64
            lpiece_size_y = 64
        if self.data_num == 1:
            # HARVARD
            piece_size_x = 520
            piece_size_y = 696
            lpiece_size_x = 65
            lpiece_size_y = 87

        if self.data_num == 2:
            # UH2
            piece_size_x = 800
            piece_size_y = 600
            lpiece_size_x = 40
            lpiece_size_y = 30

        if self.data_num == 3:
            # CAVE2
            piece_size_x = 512
            piece_size_y = 512
            lpiece_size_x = 93
            lpiece_size_y = 93

        self.buildGraph(piece_size_x, piece_size_y, lpiece_size_x, lpiece_size_y, self.rank, self.stages)
        self.saver = tf.train.Saver()

        out = {}
        average_out = {'cc': 0, 'psnr': 0, 'sam': 0, 'ssim': 0, 'rmse': 0, 'egras': 0, 'uiqi': 0}

        b = 1
        h = self.test_height
        w = self.test_width

        if h % piece_size_x != 0 or w % piece_size_y != 0:
            piece_count = (h // piece_size_x + 1) * (w // piece_size_y + 1)
        else:
            piece_count = (h // piece_size_x) * (w // piece_size_y)

        input_pieces1 = np.zeros(
            [piece_count, lpiece_size_x, lpiece_size_y, self.channels],
            dtype=np.float32)
        # print(input_pieces1.shape)
        input_pieces2 = np.zeros([piece_count, piece_size_x, piece_size_y, self.mschannels], dtype=np.float32)
        input_pieces3 = np.zeros([piece_count, self.channels, self.rank], dtype=np.float32)

        # print(input_pieces1.shape)
        data = np.zeros([h, w, self.channels], dtype=np.float32)
        checkFile(self.output_save_path)
        test_start = self.test_start
        test_end = self.test_end

        if self.data_num in [0, 1, 3]:

            with self.session as sess:
                # tf.get_variable_scope().reuse_variables()
                latest_model = tf.train.get_checkpoint_state(self.model_save_path)
                self.saver.restore(sess, latest_model.model_checkpoint_path)
                # self.saver.restore(sess,self.model_save_path+'-81')

                for i in range(test_start, test_end + 1):
                    mat = sio.loadmat(self.test_data_label_path + '%d.mat' % i)
                    # data = mat['XESES']
                    # print(mat.keys())
                    X = mat['label']
                    Y = mat['Y']
                    Z = mat['Z']
                    self.helplist.clear()
                    count = 0
                    icount = 0
                    for x in range(0, h, piece_size_x):
                        for y in range(0, w, piece_size_y):
                            if x + piece_size_x > h:
                                x = h - piece_size_x
                            if y + piece_size_y > w:
                                y = w - piece_size_y
                            # Y_piece = Y[int(x / self.ratio):int((x + piece_size_x) / self.ratio + 0.5),
                            #           int(y / self.ratio):int((y + piece_size_y) / self.ratio + 0.5), :]
                            # Y_piece = cv2.resize(Y_piece, (lpiece_size_y, lpiece_size_x), interpolation=cv2.INTER_NEAREST)
                            input_pieces1[count, :, :, :] = Y[int(x / self.ratio):int(
                                (x + piece_size_x) / self.ratio + 0.5),
                                                            int(y / self.ratio):int(
                                                                (y + piece_size_y) / self.ratio + 0.5), :]
                            input_pieces2[count, :, :, :] = Z[x:x + piece_size_x, y:y + piece_size_y, :]

                            Y_shaped = np.reshape(input_pieces1[count, :, :, :], [-1, self.channels], order='F')
                            _, _, v = np.linalg.svd(Y_shaped, full_matrices=False)
                            test_P = v.T[:, :self.rank]
                            input_pieces3[count, :, :] = test_P
                            count += 1
                    while count >= b:
                        output = sess.run(self.output,
                                          feed_dict={self.data: input_pieces1[icount * b:icount * b + b, :, :, :],
                                                     self.ms_data: input_pieces2[icount * b:icount * b + b, :, :, :],
                                                     self.P: input_pieces3[icount * b:icount * b + b, :, :]})
                        self.helplist.append(output)
                        count -= b
                        icount += 1
                    if count > 0:
                        output = sess.run(self.output,
                                          feed_dict={self.data: input_pieces1[icount * b:icount * b + count, :, :, :],
                                                     self.ms_data: input_pieces2[icount * b:icount * b + count, :, :,
                                                                   :],
                                                     self.P: input_pieces3[icount * b:icount * b + count, :, :]})
                        self.helplist.append(output)
                    input_pieces = np.concatenate(self.helplist, axis=0)
                    count = 0
                    for x in range(0, h, piece_size_x):
                        for y in range(0, w, piece_size_y):
                            if x + piece_size_x > h:
                                x = h - piece_size_x
                            if y + piece_size_y > w:
                                y = w - piece_size_y
                            data[x:x + piece_size_x, y:y + piece_size_y, :] = input_pieces[count, :, :, :]
                            count += 1

                    output = data
                    output[output < 0] = 0
                    output[output > 1] = 1.0
                    quality_reference_accessment(out, X, output, self.ratio)
                    for key in out.keys():
                        average_out[key] += out[key]
                    print('%d has finished' % i)

            for key in average_out.keys():
                average_out[key] /= (test_end - test_start + 1)

            print(average_out)
            end = time.perf_counter()
            print('用时%ss' % ((end - start) / (test_end - test_start + 1)))
        elif self.data_num in [2]:
            # UH and WV-2
            # print('invaild dataset')
            with self.session as sess:
                latest_model = tf.train.get_checkpoint_state(self.model_save_path)
                self.saver.restore(sess, latest_model.model_checkpoint_path)
                # self.saver.restore(sess,self.model_save_path+'-81')

                for i in range(test_start, test_end + 1):
                    mat = sio.loadmat(self.test_data_label_path + '%d.mat' % i)
                    Y = mat['Y']
                    Z = mat['Z']
                    self.helplist.clear()
                    count = 0
                    icount = 0
                    for x in range(0, h, piece_size_x):
                        for y in range(0, w, piece_size_y):
                            if x + piece_size_x > h:
                                x = h - piece_size_x
                            if y + piece_size_y > w:
                                y = w - piece_size_y
                            input_pieces1[count, :, :, :] = Y[int(x / self.ratio):int((x + piece_size_x) / self.ratio),
                                                            int(y / self.ratio):int((y + piece_size_y) / self.ratio), :]
                            input_pieces2[count, :, :, :] = Z[x:x + piece_size_x, y:y + piece_size_y, :]
                            Y_shaped = np.reshape(input_pieces1[count, :, :, :], [-1, self.channels], order='F')
                            _, _, v = np.linalg.svd(Y_shaped, full_matrices=False)
                            test_P = v.T[:, :self.rank]
                            input_pieces3[count, :, :] = test_P
                            count += 1
                    while count >= b:
                        output = sess.run(self.output,
                                          feed_dict={self.data: input_pieces1[icount * b:icount * b + b, :, :, :],
                                                     self.ms_data: input_pieces2[icount * b:icount * b + b, :, :, :],
                                                     self.P: input_pieces3[icount * b:icount * b + b, :, :]})
                        self.helplist.append(output)
                        count -= b
                        icount += 1
                    if count > 0:
                        output = sess.run(self.output,
                                          feed_dict={self.data: input_pieces1[icount * b:icount * b + count, :, :, :],
                                                     self.ms_data: input_pieces2[icount * b:icount * b + count, :, :,
                                                                   :],
                                                     self.P: input_pieces3[icount * b:icount * b + count, :, :]})
                        self.helplist.append(output)
                    input_pieces = np.concatenate(self.helplist, axis=0)
                    count = 0
                    for x in range(0, h, piece_size_x):
                        for y in range(0, w, piece_size_y):
                            if x + piece_size_x > h:
                                x = h - piece_size_x
                            if y + piece_size_y > w:
                                y = w - piece_size_y
                            data[x:x + piece_size_x, y:y + piece_size_y, :] = input_pieces[count, :, :, :]
                            count += 1

                    output = data
                    output[output < 0] = 0
                    output[output > 1] = 1.0
                    # sio.savemat(self.output_save_path + '%d.mat' % i, {'F': output})
                    print('%d has finished' % i)

                    # output the RGB images
                    r = 45
                    g = 20
                    b = 5
                    rgb = np.stack([output[:, :, r], output[:, :, g], output[:, :, b]], axis=-1)
                    plt.imsave(self.output_save_path + '%d.png' % i, rgb, dpi=600)

        else:
            print('invalid datasets')
            return


if __name__ == '__main__':
    start = time.perf_counter()

    network = deepNet(0)
    # network.train()
    network.test()

    end = time.perf_counter()
    print('用时%ss' % (end - start))
