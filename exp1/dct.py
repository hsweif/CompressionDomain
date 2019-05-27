import cv2
import numpy as np
from scipy import fftpack
from enum import Enum, unique
import multiprocessing as mp
from tqdm import tqdm

img_dir = '../resource/'
output_dir = img_dir+'output/'
CORE_NUM = 4

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

def solve_1d_dct(img, order):
    shape = img.shape
    h, w = shape[0], shape[1]
    res = np.zeros(shape)
    N = h if order == Order.row else w
    # FIXME: 没有整除的情况需要特判
    num_per_proc = int(N/CORE_NUM)
    proc_list = []
    manager = mp.Manager()
    res_list = manager.list()
    for i in range(0, CORE_NUM):
        start = i*num_per_proc
        end = (i+1)*num_per_proc
        proc = mp.Process(target=cal_1d_dct, args=(res_list,img,order,start,end,))
        proc.start()
        proc_list.append(proc)
    for proc in proc_list:
        proc.join()
    for start, end, proc_res in res_list:
        if order == Order.row:
           res[start:end,:] = proc_res
        else:
           res[:,start:end] = proc_res
    return res

def cal_1d_dct(res_list, img, order, start, end):
    shape = img.shape
    res = np.zeros(shape)
    h, w = shape[0], shape[1]
    for i in range(start, end):
        print(i)
        if order == Order.row:
            img_slice = np.reshape(img[i,:], (w,1))
            for v in range(0, w):
                res[i,v] = F_1d(img_slice, v, w)
        elif order == Order.col:
            img_slice = np.reshape(img[:,i], (h,1))
            for u in range(0, h):
                res[u,i] = F_1d(img_slice, u, h)
    if order == Order.row:
        res_list.append((start,end,res[start:end,:]))
    else:
        res_list.append((start,end,res[:,start:end]))


def two_dim_dct(img, block_sz=8):
    shape = img.shape
    h, w = shape[0], shape[1]
    assert(h == w)
    res = np.zeros(shape)
    block_res = np.zeros((block_sz, block_sz))
    n = int(h/block_sz)
    for ni in range(0, n):
        print(f'Iter: {ni}')
        for nj in range(0, n):
            base_img = img[ni*block_sz:(ni+1)*block_sz, nj*block_sz:(nj+1)*block_sz]
            # for u in range(0, block_sz):
            #     for v in range(0, block_sz):
            #         block_res[u, v] = F_2d(base_img, u, v, block_sz)
            block_res = F_2d(base_img, block_sz)
            res[ni*block_sz:(ni+1)*block_sz, nj*block_sz:(nj+1)*block_sz] = block_res
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

def F_2d(img, block_sz=8):
    A = np.zeros((block_sz, block_sz))
    for i in range(0, block_sz):
        for j in range(0, block_sz):
            A[i,j] = np.cos((2*j+1)*np.pi*i/(2*block_sz))
    res = np.dot(np.dot(A,img),np.transpose(A))/4
    res[0,0] = res[0,0] * c_2d(0,0)
    # A = np.zeros((1,block_sz))
    # for i in range(0, block_sz):
    #     A[0,i] = np.cos((2*i+1)*np.pi*u/(2*block_sz))
    # res = c_2d(u,v)*np.dot(np.dot(A,img),np.transpose(A))/4
    # res = 0
    # for i in range(0, block_sz):
    #     for j in range(0, block_sz):
    #         res += img[i,j]*np.cos((2*i+1)*np.pi*u/(2*block_sz))*np.cos((2*j+1)*np.pi*v/(2*block_sz))
    # res = res*c_2d(u,v)/4
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
    # res_1d = solve_1d_dct(img, Order.row)
    # cv2.imwrite(output_dir+'1d_out_1.bmp', res_1d)
    # res_1d = solve_1d_dct(res_1d, Order.col)
    # cv2.imwrite(output_dir+'1d_out.bmp', res_1d)
    N = img.shape[0]
    for sz in [N]:
        res_2d = two_dim_dct(img,sz)
        cv2.imwrite(output_dir+f'2d_out_{sz}.bmp', res_2d)
