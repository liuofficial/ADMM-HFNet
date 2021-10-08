import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.io as sio
import cv2
import scipy
import os


# this is a tool part serving for other codes

def Fill_B(B, h, w):
    '''
    change the size of Blur matrix B to meet the blur operation
    since the size of estimate B is 10*10
    :param B:
    :param h:
    :param w:
    :return:
    '''
    tB = np.zeros([h, w], dtype=np.float32)
    tB[-4:, -4:] = B[-4:, -4:]
    tB[:6, -4:] = B[:6, -4:]
    tB[-4:, :6] = B[-4:, :6]
    tB[:6, :6] = B[:6, :6]
    return tB


def generateRandomList(numlist: list, maxNum, count):
    '''
    produce needed random list
    :param numlist: random list
    :param maxNum: the max number
    :param count: the count
    :return:
    '''
    i = 0
    while i < count:
        num = random.randint(1, maxNum)
        if num not in numlist:
            numlist.append(num)
            i += 1


def unfold(x, dim):
    '''
    Matrix shift by dimension
    :param x:
    :param dim:
    :return:
    '''
    new_first_dim = x.shape[dim]
    x = np.swapaxes(x, 0, dim)
    return np.reshape(x, [new_first_dim, -1])


def getDenoiseV(X, k):
    '''
    get needed Vector by SVD for denoising
    :param X:
    :param k:
    :return:
    '''
    h, w, c = X.shape
    # _, c = X.shape
    X = np.reshape(X, [h * w, -1], order='F').T
    _, s, V = scipy.linalg.svd(X.T, full_matrices=False)
    return V.T[:, :k]


def denoise(X, V):
    '''
    denoising
    :param X:
    :param V:
    :return:
    '''
    h, w, c = X.shape
    X_t = unfold(X, 2)
    X_t_denoise = np.dot(np.dot(V, V.T), X_t)
    X = np.reshape(X_t_denoise.T, [h, w, -1], order='F')
    X[X < 0] = 0.0
    X[X > 1] = 1.0
    return X


def getFalseColorImage(X2, r=4, g=2, b=1):
    '''
    get falseColor image
    :param X2:
    :param r:
    :param g:
    :param b:
    :return:
    '''
    h, w, _ = X2.shape
    xin_rgb = np.zeros([h, w, 3], dtype=np.float32)
    xin_rgb[:, :, 0] = X2[:, :, r]
    xin_rgb[:, :, 1] = X2[:, :, g]
    xin_rgb[:, :, 2] = X2[:, :, b]
    return xin_rgb


def matchSensorWithWaveRange(psf, minWL, maxWL):
    '''
    When generating R, the band range of the sensor and image is pre-matched
    and the spectral response function of the specified range is returned
    :param psf:
    :param minWL:
    :param maxWL:
    :return:
    '''
    r, c = psf.shape
    count = 0
    for i in range(r):
        if psf[i, 0] >= minWL and psf[i, 0] <= maxWL:
            count += 1
            if count == 1:
                valid_spf = psf[i, :]
            else:
                valid_spf = np.vstack((valid_spf, psf[i, :]))
        if psf[i, 0] > maxWL:
            break
    return valid_spf


def sumtoOne(R):
    '''
    R needs to be normalized, which is divided by the sum of the rows (the number of bands in hrhs)
    :param R:
    :return:
    '''
    div = np.sum(R, axis=1)
    div = np.expand_dims(div, axis=-1)
    R = R / div
    return R

def Fill_B(B, kernel_size, h, w):
    '''
    change the size of Blur matrix B to meet the blur operation
    since the size of estimate B is r*r
    :param B:
    :param h:
    :param w:
    :return:
    '''
    tB = np.zeros([h, w], dtype=np.complex)
    tB[:kernel_size, :kernel_size] = B
    # Cyclic shift, so that the Gaussian core is located at the four corners
    tB = np.roll(np.roll(tB, -kernel_size // 2, axis=0), -kernel_size // 2, axis=1)
    return tB



def interplotedPointsToPSF(psf, bands, msbands):
    '''
    Cubic spline interpolation to obtain the required R (in the case of insufficient points)
    :param psf:
    :param bands:
    :param msbands:
    :return:
    '''
    psf_L = psf.shape[0]
    x = np.linspace(0, psf_L, psf_L)
    # print(x)
    xx = np.linspace(0, psf_L, bands)
    R = np.zeros([msbands, bands], dtype=np.float32)
    for i in range(msbands):
        # R[i, :] = make_interp_spline(x, psf[:, i + 1], xx)
        f = interp1d(x, psf[:, i + 1], kind='cubic')
        R[i, :] = f(xx)
    return sumtoOne(R)


def IKONOS_PSF():
    '''
    this is an example
    using IKONOS to get R of Pavia university
    :return:
    '''
    spf = sio.loadmat(r"f:/Fairy/hehe/ikonos_spec_resp.mat")['ikonos_sp']
    r, c = spf.shape
    count = 0
    for i in range(r):
        if spf[i, 0] >= 430 and spf[i, 0] <= 860:
            count += 1
            if count == 1:
                valid_spf = spf[i, :]
            else:
                valid_spf = np.vstack((valid_spf, spf[i, :]))
        if spf[i, 0] > 860:
            break
    # print(valid_spf)
    no_wa = valid_spf.shape[0]
    print(no_wa)
    # x = np.array([i for i in range(0, no_wa)])
    x = np.linspace(0, no_wa, no_wa)
    # print(x)
    xx = np.linspace(0, no_wa, 103)
    L = 103
    R = np.zeros([5, 103], dtype=np.float32)
    for i in range(5):
        # R[i, :] = spline(x, valid_spf[:, i + 1], xx)
        f = interp1d(x, valid_spf[:, i + 1], kind='cubic')
        # print(f)
        R[i, :] = f(xx)
    plt.plot(xx, R[2, :], color='b', label='b')
    plt.plot(xx, R[3, :], color='g', label='g')
    plt.plot(xx, R[4, :], color='r', label='r')
    plt.plot(x, valid_spf[:, 5], color='c', label='c')
    plt.plot(x, valid_spf[:, 4], color='m', label='m')
    plt.plot(x, valid_spf[:, 3], color='y', label='y')
    plt.legend()
    plt.show()


def calculateEpoch():
    '''
    calculate the epochs of TONWMD on CAVE data
    :return:
    '''
    start_lr = 2e-3
    end_lr = 1e-7
    lr = start_lr
    batch_size = 64
    total_num = 15376
    epochs = 0
    current_epoch = 0
    step = 0
    epochs_decay = 10
    while lr > end_lr:
        step += 1
        if batch_size * step >= total_num:
            current_epoch += 1
            epochs += 1
            if current_epoch >= epochs_decay:
                current_epoch = 0
                lr *= 0.5
    return epochs


def standard(X):
    '''
    Standardization
    universal
    :param X:
    :return:
    '''
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    return np.float32(X)


def checkFile(path):
    '''
    if filepath not exist make it
    :param path:
    :return:
    '''
    if not os.path.exists(path):
        os.makedirs(path)


def standard_by_norminf(X):
    '''
    Standardized by channel
    :param X:
    :return:
    '''
    _, _, c = X.shape
    for i in range(c):
        X[:, :, i] = standard(X[:, :, i])
    return np.float32(X)


def roundNum(X):
    '''
    rounding
    :param X:
    :return:
    '''
    return int(X + 0.5)


if __name__ == '__main__':
    pass
