from exp1 import dct, psnr, util
import numpy as np
import cv2

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

def quantization(img, block_sz, policy):
    assert(block_sz == 8)
    q_matrix = jpeg_qmatrix(block_sz)
    if policy == util.Policy.quan:
        med_res = img/q_matrix
        res = np.rint(med_res)
    else:
        res = img*q_matrix
    return res

def block_quant(partitions, policy, block_sz, n):
    quan_res = np.ndarray((partitions.shape))
    for ni in range(0, n):
        for nj in range(0, n):
            quan_res[ni,nj] = quantization(partitions[ni,nj],block_sz,policy)
    return quan_res

def compose(partitions, block_sz, n):
    res = np.zeros((n*block_sz, n*block_sz))
    for ni in range(0, n):
        for nj in range(0, n):
            res[ni*block_sz:(ni+1)*block_sz, nj*block_sz:(nj+1)*block_sz] = partitions[ni,nj]
    return res

if __name__ == '__main__':
    lena_img = util.open_image('lena.bmp')
    block_size = 8
    partitions, n = image_partition(lena_img, block_size)
    block_dct = block_transform(partitions, util.Policy.dct_2d, block_size, n)
    block_quan = block_quant(block_dct, util.Policy.quan, block_size, n)
    block_dequan = block_quant(block_quan, util.Policy.dequan, block_size, n)
    block_idct = block_transform(block_dequan, util.Policy.idct_2d, block_size, n)
    res_img = compose(block_idct, block_size, n)
    cv2.imwrite(util.output_dir+'quan_idct.bmp', res_img)
