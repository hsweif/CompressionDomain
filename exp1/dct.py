import cv2
import numpy as np
from scipy import fftpack
from enum import Enum, unique

img_dir = '../resource/'
output_dir = img_dir+'output/'

@unique
class Policy(Enum):
    dct_1d = 1
    dct_2d = 2

@unique
class Order(Enum):
    row = 0
    col = 1

def open_image(img_name, gray=True):
    img = cv2.imread(img_dir+img_name)
    if gray == True:
        img = color2Gray(img)
    return img

def color2Gray(img):
    #TODO: 转换成灰度要自己写吗？
    res = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return res

def one_dim_dct(img, order):
    shape = img.shape
    h, w = shape[0], shape[1]
    if order == Order.row:
        half_w = int(w/2)
        res = np.zeros((h,half_w))
        for u in range(0, h):
            print(u)
            img_slice = np.reshape(img[u,:], (w,1))
            for v in range(0, half_w):
                res[u,v] = F_1d(img_slice, v, w)
    elif order == Order.col:
        half_h = int(h/2)
        res = np.zeros((half_h,w))
        for v in range(0, w):
            print(v)
            img_slice = np.reshape(img[:,v], (h,1))
            for u in range(0, half_h):
                res[u,v] = F_1d(img_slice, u, h)
    return res

def two_dim_dct(img):
    shape = img.shape
    h, w = shape[0], shape[1]
    res = np.zeros(shape)
    for u in range(0, h):
        print(u)
        for v in range(0, w):
            res[u,v] = F_2d(img, u, v, 8)
    return res

def one_dim_idct(dct_res):
    shape = dct_res.shape
    N = shape[0]
    res = np.zeros(shape)
    print('1D IDCT')
    for u in range(0, N):
        pass


def img_1d(img, u):
    return img[u]

def img_2d(img, u, v):
    return img[u,v]

def F_1d(img, u, N):
    z = 0
    pi_u = np.pi*u
    two_N = 1/(2*N)
    for i in range(0, int(N/2)):
        z += img_1d(img,i)*np.cos(pi_u*(2*i+1)*two_N)
    res = np.sqrt(2/N)*c_1d(u)*z
    return res

def F_2d(img, u, v, block_sz=8):
    res = 0
    for i in range(0, block_sz):
        for j in range(0, block_sz):
            res += img_2d(img,i,j)*np.cos((2*i+1)*u*np.pi/16)*np.cos((2*j+1)*v*np.pi/16)
    res = (1/4)*c_2d(u,v)*res
    return res

def f_1d(dct_res, u, N):
    z = 0
    for i in range(0, N):
        z += c_1d(i)*dct_res[i]*np.cos((np.pi*(2*u+1)*i)/(2*N))
    res = np.sqrt(2/N)*z
    return res

def c_1d(u):
    return 1 if u != 0 else np.sqrt(2)/2

def c_2d(u,v):
    return 1 if u != 0 or v != 0 else np.sqrt(2)/2


def baseline(img, func_type):
    shape = img.shape
    N = shape[0]*shape[1]
    r_shape = (N, 1)
    img_reshape = np.reshape(img, r_shape)
    if func_type == Policy.dct_1d:
        res = fftpack.dct(img_reshape)
        res = np.reshape(res, shape)
    return res

if __name__ == '__main__':
    img = open_image('lena.bmp')
    res_1d = one_dim_dct(img, Order.row)
    cv2.imwrite(output_dir+'1d_out_1.bmp', res_1d)
    res_1d = one_dim_dct(res_1d, Order.col)
    cv2.imwrite(output_dir+'1d_out.bmp', res_1d)
    # res_2d = two_dim_dct(img)
    # cv2.imwrite(output_dir+'2d_out.bmp', res_2d)
