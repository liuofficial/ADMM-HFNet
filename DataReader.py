import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2
import scipy.signal as sg
import os
import shutil
import random

from functions import roundNum, standard, checkFile, unfold, generateRandomList, sumtoOne, matchSensorWithWaveRange, \
    interplotedPointsToPSF, Fill_B


# this part for data reading and patching
# data downsampling and upsampling


def getSpectralResponse():
    '''
    spectral response function for CAVE and HARVARD
    :return:
    '''
    R = np.array(
        [[2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
         [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 11, 16, 19, 21, 20, 18, 16, 14, 11, 7, 5, 3, 2, 2, 1, 1, 2, 2, 2, 2, 2],
         [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16, 9, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    div = np.sum(R, axis=1)
    div = np.expand_dims(div, axis=-1)
    R = R / div
    return R


def getBlurMatrix(kernal_size, sigma):
    '''
    get Blur matrix B
    :param kernal_size:
    :param sigma:
    :return:
    '''
    side = cv2.getGaussianKernel(kernal_size, sigma)
    Blur = np.multiply(side, side.T)
    return Blur


def get_kernal(kernal_size, sigma, rows, cols):
    '''
    Generate a Gaussian kernel and make a fast Fourier transform
    :param kernal_size:
    :param sigma:
    :return:
    '''
    # Generate 2D Gaussian filter
    blur = cv2.getGaussianKernel(kernal_size, sigma) * cv2.getGaussianKernel(kernal_size, sigma).T
    psf = np.zeros([rows, cols])
    psf[:kernal_size, :kernal_size] = blur
    # Cyclic shift, so that the Gaussian core is located at the four corners
    B1 = np.roll(np.roll(psf, -kernal_size // 2, axis=0), -kernal_size // 2, axis=1)
    # Fast Fourier Transform
    fft_b = np.fft.fft2(B1)
    # return fft_b
    return fft_b


def spectralDegrade(X, R, addNoise=True, SNR=40):
    '''
    spectral downsample
    :param X:
    :param R:
    :return:
    '''
    height, width, bands = X.shape
    X = np.reshape(X, [-1, bands], order='F')
    Z = np.dot(X, R.T)
    Z = np.reshape(Z, [height, width, -1], order='F')

    if addNoise:
        h, w, c = Z.shape
        numl = h * w * c
        sigmah = np.sqrt(np.sum(np.square(Z)) / pow(10, SNR / 10) / numl)
        print(sigmah)
        Z += sigmah * np.random.randn(h, w, c)

    return Z


def Blurs(X, B, addNoise=True, SNR=30):
    '''
    downsample using fft
    :param X:
    :param B:
    :return:
    '''
    B = np.expand_dims(B, axis=-1)
    Y = np.real(np.fft.ifftn(np.fft.fftn(X) * B))

    if addNoise:
        h, w, c = Y.shape
        numl = h * w * c
        sigmah = np.sqrt(np.sum(np.square(Y)) / pow(10, SNR / 10) / numl)
        print(sigmah)
        Y += sigmah * np.random.randn(h, w, c)
    return Y


def upSample(X, ratio=8):
    '''
    upsample using cubic
    :param X:
    :param ratio:
    :return:
    '''
    h, w, c = X.shape
    return cv2.resize(X, (w * ratio, h * ratio), interpolation=cv2.INTER_CUBIC)


def downSample(X, B, ratio, addNoise=True, SNR=30):
    '''
    downsample using fft
    :param X:
    :param B:
    :param ratio:
    :return:
    '''
    B = np.expand_dims(B, axis=-1)
    Y = np.real(np.fft.ifftn(np.fft.fftn(X) * B))

    if addNoise:
        h, w, c = Y.shape
        numl = h * w * c
        sigmah = np.sqrt(np.sum(np.square(Y)) / pow(10, SNR / 10) / numl)
        print(sigmah)
        Y += sigmah * np.random.randn(h, w, c)

    # downsample
    Y = Y[::ratio, ::ratio, :]
    return Y


def readCAVEData(path, mat_path):
    '''
    Read initial CAVE data
    since the original data is standardized we do not repeat it
    :return:
    '''
    hsi = np.zeros([512, 512, 31], dtype=np.float32)
    checkFile(mat_path)
    count = 0
    for dir in os.listdir(path):
        concrete_path = path + '/' + dir + '/' + dir
        for i in range(31):
            fix = str(i + 1)
            if i + 1 < 10:
                fix = '0' + str(i + 1)
            png_path = concrete_path + '/' + dir + '_' + fix + '.png'
            try:
                hsi[:, :, i] = plt.imread(png_path)
            except:
                img = plt.imread(png_path)
                img = img[:, :, :3]
                img = np.mean(img, axis=2)
                hsi[:, :, i] = img

        count += 1
        print('%d has finished' % count)
        sio.savemat(mat_path + str(count) + '.mat', {'HS': hsi})


def readHarvardData(path, mat_path):
    '''
    Read the initial HARVARD data
    data need to be standardized
    :param path
    :param mat_path
    :return:
    '''
    checkFile(mat_path)
    count = 0
    for dir in os.listdir(path):
        count += 1
        hs = sio.loadmat(path + '/' + dir)['ref']
        hs = standard(hs)
        sio.savemat(mat_path + "%d.mat" % count, {'HS': hs})
        print('%d has finished' % count)


def createSimulateData(data_index, B, R, ratio, h=93, w=93):
    '''
    create simulated data
    :param data_index:
    :param B:
    :param R:
    :param ratio
    :return:
    '''
    if data_index == 0:
        # CAVE
        mat_path = r'CAVEMAT/'
        num_start = 1
        num_end = 32

    elif data_index == 1:
        # Harvard
        mat_path = r'HARVARDMAT/'
        num_start = 1
        num_end = 50
    elif data_index == 3:
        # CAVE2
        mat_path = r'CAVEMAT2/'
        num_start = 1
        num_end = 32

    if data_index in [0, 1]:
        for i in range(num_start, num_end + 1):
            mat = sio.loadmat(mat_path + '%d.mat' % i)
            hs = mat['HS']
            ms = spectralDegrade(hs, R, True)  # generate the HR-MSI
            lrhs = downSample(hs, B, ratio, True)  # generate the LR-HSI
            sio.savemat(mat_path + str(i) + '.mat', {'label': hs, 'Z': ms, 'Y': lrhs})
            print('%d has finished' % i)

    elif data_index == 3:
        for i in range(num_start, num_end + 1):
            mat = sio.loadmat(mat_path + '%d.mat' % i)
            hs = mat['HS']
            ms = spectralDegrade(hs, R, True)  # generate the HR-MSI
            lrhs = Blurs(hs, B)
            lrhs = cv2.resize(lrhs, (int(w / ratio + 0.5), int(h / ratio + 0.5)),
                              interpolation=cv2.INTER_NEAREST)  # generate the LR-HSI
            sio.savemat(mat_path + str(i) + '.mat', {'label': hs, 'Z': ms, 'Y': lrhs})
            print('%d has finished' % i)


def cutTrainingPiecesForSimulatedDataset(data_index):
    '''
    produce training pieces
    :param train_index:
    :return:
    '''
    if data_index == 0:
        # CAVE
        # the first 20 images are patched for training and verifying
        piece_size = 64
        rank = 20  # number of spectral basis
        stride = 16
        rows, cols = 512, 512
        num_start = 1
        num_end = 20
        mat_path = r'CAVEMAT/'
        count = 0
        bands = 31
        ratio = 8
        piece_save_path = mat_path + 'pieces/normal/train/'

    elif data_index == 1:
        # Harvard
        # the first 30 images are patched for training and verifying
        piece_size = 64
        rank = 20  # number of spectral basis
        stride = 16
        rows, cols = 1040, 1392
        num_start = 1
        num_end = 30
        mat_path = r'HARVARDMAT/'
        count = 0
        bands = 31
        ratio = 8
        piece_save_path = mat_path + 'pieces/normal/train/'

    elif data_index == 3:
        # CAVE2
        # the first 20 images are patched for training and verifying
        piece_size = 64
        rank = 20  # number of spectral basis
        stride = 16
        rows, cols = 512, 512
        num_start = 1
        num_end = 20
        mat_path = r'CAVEMAT2/'
        count = 0
        bands = 31
        ratio = 5.5
        lpiece_size = int(piece_size / ratio + 0.5)
        piece_save_path = mat_path + 'pieces/normal/train/'

    checkFile(piece_save_path)
    if data_index in [0, 1]:
        for i in range(num_start, num_end + 1):
            mat = sio.loadmat(mat_path + '%d.mat' % i)
            X = mat['label']
            Y = mat['Y']  # The upsampled image of Y using bicubic
            Z = mat['Z']
            for x in range(0, rows - piece_size + stride, stride):
                for y in range(0, cols - piece_size + stride, stride):
                    Y_piece = Y[x // ratio:(x + piece_size) // ratio, y // ratio:(y + piece_size) // ratio, :]
                    Z_piece = Z[x:x + piece_size, y:y + piece_size, :]
                    label_piece = X[x:x + piece_size, y:y + piece_size, :]

                    Y_shaped = np.reshape(Y_piece, [-1, bands], order='F')
                    U, D, V = np.linalg.svd(Y_shaped, full_matrices=False)
                    P = V.T[:, :rank]
                    sio.savemat(piece_save_path + 'a%d.mat' % count,
                                {'P': P, 'Y': Y_piece, 'Z': Z_piece, 'label': label_piece})
                    count += 1
                    print('piece num %d has saved' % count)
            print('%d has finished' % i)
    elif data_index == 3:
        for i in range(num_start, num_end + 1):
            mat = sio.loadmat(mat_path + '%d.mat' % i)
            X = mat['label']
            Y = mat['Y']  # The upsampled image of Y using bicubic
            Z = mat['Z']
            for x in range(0, rows - piece_size + stride, stride):
                for y in range(0, cols - piece_size + stride, stride):
                    Y_piece = Y[int(x / ratio):int((x + piece_size) / ratio + 0.5),
                              int(y / ratio):int((y + piece_size) / ratio + 0.5), :]
                    Y_piece = cv2.resize(Y_piece, (lpiece_size, lpiece_size), interpolation=cv2.INTER_NEAREST)
                    Z_piece = Z[x:x + piece_size, y:y + piece_size, :]
                    label_piece = X[x:x + piece_size, y:y + piece_size, :]

                    Y_shaped = np.reshape(Y_piece, [-1, bands], order='F')
                    U, D, V = np.linalg.svd(Y_shaped, full_matrices=False)
                    P = V.T[:, :rank]
                    sio.savemat(piece_save_path + 'a%d.mat' % count,
                                {'P': P, 'Y': Y_piece, 'Z': Z_piece, 'label': label_piece})
                    count += 1
                    print('piece num %d has saved' % count)
            print('%d has finished' % i)
    print('done')
    return count


def cutUHPieces(mat_path, B):
    # import math
    piece_size = 60
    stride_x = 10
    stride_y = 10
    ratio = 20
    rows, cols = 800, 600
    num_start = 1
    num_end = 5
    bands = 48
    piece_save_path = mat_path + r'pieces\normal\train\\'
    checkFile(piece_save_path)
    count = 0
    rank = 4

    B2 = Fill_B(B, ratio, rows, cols)
    B2 = np.fft.fft2(B2)

    for k in range(num_start, num_end + 1):
        mat = sio.loadmat(mat_path + '%d.mat' % k)
        # X = mat['HS']
        Y = mat['Y']
        Z = mat['Z']

        X = []
        Zs = []
        for i in range(ratio):
            X.append([])
            Zs.append([])

        Z = downSample(Z, B2, ratio, False)
        # Z = Z[::ratio,::ratio,:]
        Z1 = Z
        Y1 = Y

        for i in range(ratio):
            for j in range(ratio):
                t = random.random()
                # print(t)
                if t > 0.5:
                    rotate = random.uniform(-1, 1)
                    angle = 45 * rotate
                    matrix = cv2.getRotationMatrix2D((30 / 2, 40 / 2), angle, 1)
                    Y = cv2.warpAffine(Y1, matrix, (30, 40))
                    Z = cv2.warpAffine(Z1, matrix, (30, 40))
                elif t > 0.2:
                    flip = random.randint(-1, 1)
                    Y = cv2.flip(Y1, flip)
                    Z = cv2.flip(Z1, flip)
                else:
                    Y = Y1
                    Z = Z1

                # print(Z.shape)
                X[i].append(Y)
                Zs[i].append(Z)
        #
        for i in range(ratio):
            X[i] = np.concatenate(X[i], axis=0)
            Zs[i] = np.concatenate(Zs[i], axis=0)

        X = np.concatenate(X, axis=1)
        Zs = np.concatenate(Zs, axis=1)
        # # print(X.shape)
        Z = Zs

        Y = downSample(X, B2, ratio, False)

        for x in range(0, rows - piece_size + stride_x, stride_x):
            for y in range(0, cols - piece_size + stride_y, stride_y):

                if x + piece_size > rows:
                    x = rows - piece_size
                if y + piece_size > cols:
                    y = cols - piece_size

                label_piece = X[x:x + piece_size, y:y + piece_size, :]
                Z_piece = Z[x:x + piece_size, y:y + piece_size, :]
                Y_piece = Y[x // ratio:(x + piece_size) // ratio, y // ratio:(y + piece_size) // ratio, :]

                count += 1
                Y_shaped = np.reshape(label_piece, [-1, bands], order='F')
                U, D, V = np.linalg.svd(Y_shaped, full_matrices=False)
                P = V.T[:, :rank]
                sio.savemat(piece_save_path + 'a%d.mat' % count,
                            {'P': P, 'Y': Y_piece, 'Z': Z_piece, 'label': label_piece})
                print('piece num %d has saved' % count)
        print('%d has finished' % k)
    return count


def generateVerticationSet(mat_path, num):
    '''
    Randomly select 20% as the verification set
    :param train_path:
    :param verti_path:
    :param num:
    :return:
    '''
    ratio = 0.2
    verti_num = int(num * ratio)
    num_list = []
    train_path = mat_path + 'pieces/normal/train/'
    verti_path = mat_path + 'pieces/normal/valid/'
    checkFile(verti_path)
    generateRandomList(num_list, num, verti_num)
    print(num_list.__len__())
    for ind, val in enumerate(num_list):
        # mat = sio.loadmat(train_path+'%d.mat'%val)
        # sio.savemat(verti_path+'%d.mat'%(ind+1),mat)
        try:
            shutil.copy(train_path + 'a%d.mat' % val, verti_path + '%d.mat' % (ind + 1))
            os.remove(train_path + 'a%d.mat' % val)
            print('%d has created' % (ind + 1))
        except:
            print('raise error')
    print('veticatication set created')
    print('done rerank train set')
    # rename the left train pieces
    # reRankfile(train_path, 'a')
    reRankfile(train_path, '')
    return num_list.__len__(), num - num_list.__len__()


def reRankfile(path, name):
    '''
    Reorder mat by renaming
    :param path:
    :param name:
    :return:
    '''
    count = 0
    file_list = os.listdir(path)
    for file in file_list:
        try:
            count += 1
            newname = str(count)
            print(newname)
            os.rename(path + file, path + '%s.mat' % (newname))
        except:
            print('error')


if __name__ == '__main__':
    # manipulating datasets including CAVE, Harvard, University of Houston, CAVE with non-integer resolution ratio
    data_index = 0  # 0, 1 ,2, 3 represents the four datasets, respectively

    if data_index == 0:
        # CAVE
        path = 'f:/Fairy/CAVE/'  # replace with your path which puts the downloading CAVE data
        mat_path = r'CAVEMAT/'  # the path of saving the entile HSI with .mat format
        # convert the data into .mat format
        readCAVEData(path, mat_path)

        # produce the simulated data according to the Wald's protocol
        B = get_kernal(8, 2, 512, 512)  # the blurring kernel with size of 8*8 and a standard deviation of 2
        # The spectral response matrix coming from Nikon Camera (400-700 nm)
        R = np.array(
            [[2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
             [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 11, 16, 19, 21, 20, 18, 16, 14, 11, 7, 5, 3, 2, 2, 1, 1, 2, 2, 2, 2, 2],
             [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16, 9, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        R = sumtoOne(R)  # to ensure the sum of each row should be 1

        ratio = 8  # the spatial resolution ratio
        createSimulateData(data_index, B, R, ratio=ratio)

        ## produce the training and verification pieces
        count = cutTrainingPiecesForSimulatedDataset(data_index)
        ## randomly select 20% of the training pieces as the verification pieces
        generateVerticationSet(mat_path, count)


    elif data_index == 1:
        # Harvard
        path = r'f:/Fairy/HARVARD/CZ_hsdb'  # replace with your path which puts the downloading Harvard data
        mat_path = r'HARVARDMAT/'  # the path of saving the entile HSI with .mat format
        # # convert the data into .mat format
        readHarvardData(path, mat_path)

        ## produce the simulated data according to the Wald's protocol
        B = get_kernal(8, 2, 1040, 1392)  # the blurring kernel with size of 8*8 and a standard deviation of 2
        # The spectral response matrix coming from Nikon Camera (400-700 nm)
        R = np.array(
            [[2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
             [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 11, 16, 19, 21, 20, 18, 16, 14, 11, 7, 5, 3, 2, 2, 1, 1, 2, 2, 2, 2, 2],
             [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16, 9, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        R = sumtoOne(R)

        # # the wavelength of Harvard is ranging from 420 to 720 nm while the Nikon camera covers 400-700nm
        # # wavelength ranges between the sensor and the images should match
        # # therefore, 420-700 nm are shared and interpolation should be adpoted
        # wavelength = np.array([[i for i in range(400, 710, 10)]])
        # srf = np.concatenate([wavelength, R], axis=0)  # add the wavelength dimension
        # srf = srf.T
        #
        # srf = matchSensorWithWaveRange(srf, 420, 700)  # match the wavelength range
        # R = interplotedPointsToPSF(srf, 31,
        #                            3)  # spread the number of points to the number of bands by the cubic spline interpolation

        ratio = 8  # the spatial resolution ratio
        createSimulateData(data_index, B, R, ratio=ratio)

        ## produce the training and verification pieces
        count = cutTrainingPiecesForSimulatedDataset(data_index)
        ## randomly select 20% of the training pieces as the verification pieces
        generateVerticationSet(mat_path, count)


    elif data_index == 2:
        # UH
        path = r'f:/Fairy/HOUSTAN/'  # replace with your path which puts the downloading UH data
        mat_path = r'HOUSTANMAT/'  # the path of saving the entile HSI with .mat format

        checkFile(mat_path)

        # We choose the left-bottom part (601 * 596 * 48) of the LR-HSI as the experimental data
        lrhs = sio.loadmat(path + 'hs.mat')['data']
        lrhs = lrhs[601:,:596,:48]

        # Correspondingly, the HR-MSI are of size 12020*11920*3, which is the no. 1 image
        ms = sio.loadmat(path + 'ms%d.mat' %1)['img']

        # Both the data need to be standardized
        lrhs = standard(lrhs)
        ms = standard(ms)

        # UH is a real dataset, so we don't have to produce the LR-HSI and HR-MSI
        # And the spatial resolution of HR-MSI is too high, for convenience, we crop five sub-images for experimenting
        # the sub-images of HR-MSI are of size 800 * 600
        for i in range(5):
            Z = ms[:800, i * 600:(i + 1) * 600, :]
            Y = lrhs[:40, i * 30:(i + 1) * 30, :]
            sio.savemat(mat_path + '%s.mat' % (i + 1), {'Y': Y, 'Z': Z})
            print('%d has finished' % (i + 1))

        # since the ground-truth is unavaiable, we choose to construct the training set according to the Wald's protocol with the estimated blurring kernel by HySure.
        B = sio.loadmat('R_B.mat')['B']  # B is estimated by HySure
        # produce the training and verification pieces
        count = cutUHPieces(mat_path, B)

        # randomly select 20% of the training pieces as the verification pieces
        generateVerticationSet(mat_path, count)

    elif data_index == 3:
        # CAVE
        path = 'f:/Fairy/CAVE/'  # replace with your path which puts the downloading CAVE data
        mat_path = r'CAVEMAT2/'  # the path of saving the entile HSI with .mat format
        # convert the data into .mat format
        readCAVEData(path, mat_path)

        # produce the simulated data according to the Wald's protocol
        B = get_kernal(5, 2, 512, 512)  # the blurring kernel with size of 5*5 and a standard deviation of 2
        # The spectral response matrix coming from Nikon Camera (400-700 nm)
        R = np.array(
            [[2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
             [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 11, 16, 19, 21, 20, 18, 16, 14, 11, 7, 5, 3, 2, 2, 1, 1, 2, 2, 2, 2, 2],
             [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16, 9, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        R = sumtoOne(R)  # to ensure the sum of each row should be 1

        ratio = 5.5  # the spatial resolution ratio
        createSimulateData(data_index, B, R, ratio=ratio, h=512, w=512)

        ## produce the training and verification pieces
        count = cutTrainingPiecesForSimulatedDataset(data_index)
        ## randomly select 20% of the training pieces as the verification pieces
        generateVerticationSet(mat_path, count)

    else:
        print('New dataset, you need add the manipulating codes following the above')
