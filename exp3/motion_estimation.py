import cv2
import numpy as np
from exp1.util import *
import exp1.dct as dct

INF = 1e10

def open_video(video_name='cars.avi'):
    video = cv2.VideoCapture(img_dir+video_name)
    return video

def cal_mse(x, y):
    delta = x-y
    delta = delta*delta
    res = np.average(delta)
    return res

def sub_frame(frame, x, y, offset):
    return x, y, frame[y:y+offset,x:x+offset]

def brute_search(prev_frame, frame, x, y, shift, offset):
    start_x, start_y = x - shift, y - shift
    min_mse = INF
    tx, ty, orig_frame = sub_frame(prev_frame,x,y,offset)
    next_x, next_y = x, y
    for xi in range(start_x, x+shift):
        for yi in range(start_y, y+shift):
            tx, ty, sf = sub_frame(frame, xi, yi, offset)
            mse = cal_mse(orig_frame, sf)
            if mse < min_mse:
                next_x, next_y = xi, yi
                min_mse = mse
    print(min_mse)
    return next_x, next_y



def tss(prev_frame, frame, x, y, shift=4, offset=16, min_mse=INF):
    if shift < 1:
        return x, y
    cand_list = []
    tmp_x, tmp_y, orig_frame = sub_frame(prev_frame,x,y,offset)
    cand_list.append(sub_frame(frame,x,y,offset))
    cand_list.append(sub_frame(frame,x-shift,y,offset))
    cand_list.append(sub_frame(frame,x+shift,y,offset))
    cand_list.append(sub_frame(frame,x,y-shift,offset))
    cand_list.append(sub_frame(frame,x,y+shift,offset))
    cand_list.append(sub_frame(frame,x+shift,y+shift,offset))
    cand_list.append(sub_frame(frame,x-shift,y+shift,offset))
    cand_list.append(sub_frame(frame,x+shift,y-shift,offset))
    cand_list.append(sub_frame(frame,x-shift,y-shift,offset))
    next_x, next_y = x, y
    for xi,yi,frame_i in cand_list:
        mse = cal_mse(orig_frame, frame_i)
        if mse < min_mse:
            next_x, next_y = xi, yi
            min_mse = mse
    return tss(prev_frame, frame, next_x, next_y, int(shift/2), offset, min_mse)


x_1, y_1 = 115, 117
cap = open_video()

ret, prev_frame = cap.read()
prev_frame = color2Gray(prev_frame)
x_2, y_2 = x_1+16, y_1+16
cv2.rectangle(prev_frame, (x_1, y_1), (x_2, y_2), (0,0,0))
cv2.imshow("capture", prev_frame)
cv2.waitKey()

while(1):
    ret, frame = cap.read()
    frame = color2Gray(frame)
    # x_1, y_1 = tss(prev_frame,frame,x_1,y_1,4)
    x_1, y_1 = brute_search(prev_frame,frame,x_1,y_1,16,16)
    x_2, y_2 = x_1+16, y_1+16
    cv2.rectangle(frame, (x_1, y_1), (x_2, y_2), (0,0,0))
    cv2.imshow("capture", frame)
    cv2.waitKey()
    prev_frame = frame
