import numpy as np
import exp1.util as util


def cal_psnr(orig_img, new_img):
    mse = cal_mse(orig_img, new_img)
    return 10 * np.log(np.power(255,2)/mse)

def cal_mse(orig_img, new_img):
    assert(orig_img.shape == new_img.shape)
    h, w = orig_img.shape[0], orig_img.shape[1]
    frame_size = h*w
    delta_img = orig_img - new_img
    res = 0
    for i in range(0, h):
        for j in range(0, w):
            res += delta_img[i,j]*delta_img[i,j]
    return res/frame_size

def get_psnr(img_list):
    psnr_dict = dict()
    orig_img = util.open_image('lena.bmp')
    for img_name in img_list:
        new_img = util.open_image(img_name)
        psnr_dict[img_name] = cal_psnr(orig_img, new_img)
    return psnr_dict

if __name__ == '__main__':
    img_list = ['output/1d_idct.bmp', 'output/2d_idct_8.bmp', 'output/2d_idct_512.bmp']
    psnr = get_psnr(img_list)
    for img in img_list:
        print(f"{img}'s PSNR: {psnr[img]}")
