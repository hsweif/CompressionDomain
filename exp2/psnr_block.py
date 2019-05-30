from exp1 import dct, psnr, util
import numpy as np
import cv2
import matplotlib.pyplot as plt

def image_partition(img, block_sz=8):
    # FIXME: Does not consider padding problem yet.
    shape = img.shape
    h, w = shape[0], shape[1]
    assert(h == w)
    n = int(h/block_sz)
    assert(n*block_sz == h)
    partitions = np.ndarray((n,n,block_sz,block_sz))
    for ni in range(0, n):
        for nj in range(0, n):
            partitions[ni,nj] = img[ni*block_sz:(ni+1)*block_sz, nj*block_sz:(nj+1)*block_sz]
    return partitions, n

def block_transform(partitions, policy, block_sz,n):
    block_res = np.ndarray(partitions.shape)
    for ni in range(0, n):
        for nj in range(0, n):
            block_res[ni,nj] = dct.two_dim_transform(partitions[ni,nj], policy, block_sz)
    return block_res

def jpeg_qmatrix(block_sz=8):
    '''
    :param block_sz: The size of quantization matrix
    :return: Quantization matrix based on JPEG standard
    '''
    # TODO: Make it more general with different block size.
    assert(block_sz==8)
    q_matrix = np.ndarray((block_sz, block_sz))
    q_matrix[0,:] = [16, 11, 10, 16, 24, 40, 51, 61]
    q_matrix[1,:] = [12, 12, 14, 19, 26, 58, 60, 55]
    q_matrix[2,:] = [14, 13, 16, 24, 40, 57, 69, 56]
    q_matrix[3,:] = [14, 17, 22, 29, 51, 87, 80, 62]
    q_matrix[4,:] = [18, 22, 37, 56, 68, 109,103,77]
    q_matrix[5,:] = [24, 35, 55, 64, 81, 104,113,92]
    q_matrix[6,:] = [49, 64, 78, 87, 103,121,120,101]
    q_matrix[7,:] = [72, 92, 95, 98, 112,100,103,99]
    return q_matrix

def quantization(img, block_sz, policy, a_cof=1):
    assert(block_sz == 8)
    q_matrix = jpeg_qmatrix(block_sz)*a_cof
    if policy == util.Policy.quan:
        med_res = img/q_matrix
        res = np.rint(med_res)
    else:
        res = img*q_matrix
    return res

def block_quant(partitions, policy, block_sz, n, a_cof):
    quan_res = np.ndarray((partitions.shape))
    for ni in range(0, n):
        for nj in range(0, n):
            quan_res[ni,nj] = quantization(partitions[ni,nj],block_sz,policy,a_cof)
    return quan_res

def solve_exp2(orig_part, dct_partitions, block_sz, n, a_cof=1.0):
    quan_res = block_quant(dct_partitions, util.Policy.quan, block_sz, n, a_cof)
    dequan_res = block_quant(quan_res, util.Policy.dequan, block_sz, n, a_cof)
    idct_block = block_transform(dequan_res, util.Policy.idct_2d, block_size, n)
    psnr_res, psnr_avg = partition_psnr(orig_part, idct_block, n)
    res_img = compose(idct_block, block_sz, n)
    return res_img, psnr_avg

def partition_psnr(orig_part, new_part, n):
    psnr_res = np.zeros((n,n))
    for ni in range(0, n):
        for nj in range(0, n):
            psnr_res[ni,nj] = psnr.cal_psnr(orig_part[ni,nj], new_part[ni,nj])
    psnr_avg = np.average(psnr_res)
    return psnr_res, psnr_avg

def compose(partitions, block_sz, n):
    res = np.zeros((n*block_sz, n*block_sz))
    for ni in range(0, n):
        for nj in range(0, n):
            res[ni*block_sz:(ni+1)*block_sz, nj*block_sz:(nj+1)*block_sz] = partitions[ni,nj]
    return res

if __name__ == '__main__':
    lena_img = util.open_image('lena.bmp')
    block_size = 8
    result = []
    psnr_list = []
    orig_img = util.open_image('lena.bmp')
    result.append(orig_img)
    partitions, n = image_partition(lena_img, block_size)
    dct_partitions = block_transform(partitions, util.Policy.dct_2d, block_size, n)
    cof_list = []

    for i in range(1, 9):
        a_cof = i/5
        cof_list.append(a_cof)
        res_img, psnr_avg = solve_exp2(partitions, dct_partitions, block_size, n, a_cof)
        print(f'Round {i} quantization, cof = {a_cof}, psnr = {psnr_avg}')
        psnr_list.append(psnr_avg)
        result.append(res_img)

    plt.figure()
    for i in range(1, 10):
        plt.subplot(3,3,i)
        if i == 1:
            plt.title('original image')
        else:
            plt.title(f'a={(i-1)/5}')
        plt.imshow(result[i-1], cmap=plt.cm.gray)
        plt.axis('off')
    plt.show()

    plt.figure()
    plt.title('PSNR - a relationship')
    plt.xlabel('a')
    plt.ylabel('PSNR')
    plt.plot(cof_list, psnr_list)
    plt.show()
