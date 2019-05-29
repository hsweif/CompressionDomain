from exp1 import dct, psnr, util
import numpy as np

def image_partition(img, block_sz=8):
    # FIXME: Does not consider padding problem yet.
    partitions = []
    shape = img.shape
    h, w = shape[0], shape[1]
    assert(h == w)
    n = int(h/block_sz)
    assert(n*block_sz == h)
    for ni in range(0, n):
        for nj in range(0, n):
            sub_img = img[ni*block_sz:(ni+1)*block_sz, nj*block_sz:(nj+1)*block_sz]
            partitions.append(sub_img)
    return partitions

def block_transform(partitions, policy, block_sz,):
    block_res = []
    for pt_img in partitions:
        pt_res = dct.two_dim_transform(pt_img, policy, block_sz)
        pt_res = quantization(pt_res, block_sz)
        block_res.append(pt_res)
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


def quantization(img, block_sz):
    assert(block_sz == 8)
    q_matrix = jpeg_qmatrix(block_sz)
    med_res = img/q_matrix
    res = np.rint(med_res)
    return res

if __name__ == '__main__':
    lena_img = util.open_image('lena.bmp')
    block_size = 8
    partitions = image_partition(lena_img, block_size)
    block_dct = block_transform(partitions, util.Policy.dct_2d, block_size)
    block_idct = block_transform(block_dct, util.Policy.idct_2d, block_size)
