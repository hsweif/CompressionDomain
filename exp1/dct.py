import numpy as np
from scipy import fftpack
import multiprocessing as mp
from exp1.util import *

CORE_NUM = 4

def solve_1d_dct(img, order, policy=Policy.idct_1d):
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
        proc = mp.Process(target=cal_1d_dct, args=(res_list,policy,img,order,start,end,))
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

def cal_1d_dct(res_list, policy, img, order, start, end):
    shape = img.shape
    res = np.zeros(shape)
    h, w = shape[0], shape[1]
    for i in range(start, end):
        if order == Order.row:
            img_slice = np.reshape(img[i,:], (w,1))
            for v in range(0, w):
                if policy == Policy.dct_1d:
                    res[i,v] = F_1d(img_slice, v, w)
                elif policy == Policy.idct_1d:
                    res[i,v] = f_1d(img_slice, v, w)
        elif order == Order.col:
            img_slice = np.reshape(img[:,i], (h,1))
            for u in range(0, h):
                if policy == Policy.dct_1d:
                    res[u,i] = F_1d(img_slice, u, h)
                elif policy == Policy.idct_1d:
                    res[u,i] = f_1d(img_slice, u, h)
    if order == Order.row:
        res_list.append((start,end,res[start:end,:]))
    else:
        res_list.append((start,end,res[:,start:end]))

def two_dim_transform(img, policy, block_sz=8):
    shape = img.shape
    h, w = shape[0], shape[1]
    assert(h == w)
    res = np.zeros(shape)
    n = int(h/block_sz)
    assert(n*block_sz == h)
    for ni in range(0, n):
        # print(f'Iter: {ni}')
        for nj in range(0, n):
            base_img = img[ni*block_sz:(ni+1)*block_sz, nj*block_sz:(nj+1)*block_sz]
            if(policy == Policy.dct_2d):
                block_res = F_2d(base_img, block_sz)
            elif(policy == Policy.idct_2d):
                block_res = f_2d(base_img, block_sz)
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
    for i in range(0, N):
        z += img_1d(img,i)*np.cos(pi_u*(2*i+1)*two_N)
    res = np.sqrt(2/N)*c_1d(u)*z
    return res

def F_2d(img, block_sz):
    A = np.zeros((block_sz, block_sz))
    for i in range(0, block_sz):
        for j in range(0, block_sz):
            A[i,j] = np.cos((2*j+1)*np.pi*i/(2*block_sz))
    res = (2/block_sz)*np.dot(np.dot(A,img),np.transpose(A))
    for i in range(0, block_sz):
        for j in range(0, block_sz):
            res[i,j] = res[i,j] * c_2d(i,j)
    return res

def f_2d(img, block_sz):
    A = np.zeros((block_sz, block_sz))
    for i in range(0, block_sz):
        for j in range(0, block_sz):
            A[i,j] = c_1d(j)*np.cos((2*i+1)*np.pi*j/(2*block_sz))
    res = (2/block_sz)*np.dot(np.dot(A,img),np.transpose(A))
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
    # if u != 0 and v != 0:
    #     return 1
    # return np.sqrt(2)/2
    if u != 0 and v != 0:
        return 1
    if u == 0 and v == 0:
        return 1/2
    return np.sqrt(2)/2

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
    # cv2.imwrite(output_dir+'gray_lena.bmp',img)
    # print('Solving 1D DCT. First row then column, solving in multiprocess')
    # print('Row DCT')
    # dct_1d_row = solve_1d_dct(img, Order.row,Policy.dct_1d)
    # print('Column DCT')
    # dct_1d = solve_1d_dct(dct_1d_row, Order.col,Policy.dct_1d)
    # cv2.imwrite(output_dir+'1d_out_row.bmp', dct_1d_row)
    # cv2.imwrite(output_dir+'1d_out.bmp', dct_1d)
    # print('Column IDCT')
    # idct_1d_col = solve_1d_dct(dct_1d, Order.col, Policy.idct_1d)
    # print('Column DCT')
    # idct_1d = solve_1d_dct(idct_1d_col, Order.row, Policy.idct_1d)
    # cv2.imwrite(output_dir+'1d_idct.bmp', idct_1d)
    print('Solving 2D DCT.')
    N = img.shape[0]
    for sz in [8,N]:
        dct_2d = two_dim_transform(img,Policy.dct_2d,sz)
        idct_2d = two_dim_transform(dct_2d, Policy.idct_2d, sz)
        cv2.imwrite(output_dir+f'2d_dct_{sz}.bmp', dct_2d)
        cv2.imwrite(output_dir+f'2d_idct_{sz}.bmp', idct_2d)
