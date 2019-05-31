import cv2
from enum import Enum, unique

img_dir = '../resource/'
output_dir = img_dir+'output/'

@unique
class Policy(Enum):
    dct_1d = 1
    dct_2d = 2
    idct_2d = 3
    idct_1d = 4
    quan = 5
    dequan = 6

@unique
class Domain(Enum):
    pixel = 1
    freq = 2

@unique
class Order(Enum):
    row = 0
    col = 1

def open_image(img_name, gray=True):
    img = cv2.imread(img_dir+img_name)
    if gray == True:
        img = color2Gray(img)
    return img

def save_image(img_name):
    cv2.imwrite(output_dir+img_name)

def color2Gray(img):
    #TODO: 转换成灰度要自己写吗？
    res = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return res
