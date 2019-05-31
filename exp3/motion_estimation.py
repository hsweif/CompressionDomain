import cv2
import numpy as np
import exp1.util as util
import exp1.dct as dct
import matplotlib.pyplot as plt

INF = 1e10

def open_video(video_name='cars.avi'):
    video = cv2.VideoCapture(util.img_dir+video_name)
    return video

def cal_mse(x, y):
    delta = x-y
    delta = delta*delta
    res = np.average(delta)
    return res

def sub_frame(frame, x, y, offset):
    return x, y, frame[y:y+offset,x:x+offset]

def brute_search(prev_frame, frame, x, y, shift, offset, domain=util.Domain.pixel):
    start_x, start_y = x - shift, y - shift
    min_mse = INF
    tx, ty, orig_frame = sub_frame(prev_frame,x,y,offset)
    next_x, next_y = x, y
    for xi in range(start_x, x+shift):
        for yi in range(start_y, y+shift):
            tx, ty, sf = sub_frame(frame, xi, yi, offset)
            if domain == util.Domain.pixel:
                mse = cal_mse(orig_frame, sf)
            else:
                mse = cal_mse(dct.two_dim_transform(orig_frame, util.Policy.dct_2d, offset), dct.two_dim_transform(sf, util.Policy.dct_2d, offset))
            if mse < min_mse:
                next_x, next_y = xi, yi
                min_mse = mse
    return next_x, next_y, min_mse

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
        # mse = cal_mse(dct.two_dim_transform(orig_frame, Policy.dct_2d, offset), dct.two_dim_transform(frame_i, Policy.dct_2d, offset))
        if mse < min_mse:
            next_x, next_y = xi, yi
            min_mse = mse
    return tss(prev_frame, frame, next_x, next_y, int(shift/2), offset, min_mse)


x_1, y_1 = 115, 117
cap = open_video()

ret, prev_frame = cap.read()
x_2, y_2 = x_1+16, y_1+16
cv2.rectangle(prev_frame, (x_1, y_1), (x_2, y_2), (0,0,0))
cv2.imshow("capture", prev_frame)
cv2.waitKey()
mse_list = []
frame_num = 25
for cnt in range(0, frame_num):
    ret, frame = cap.read()
    prev_frame_gray = util.color2Gray(prev_frame)
    frame_gray = util.color2Gray(frame)
    # x_1, y_1 = tss(prev_frame,frame,x_1,y_1,4)
    tx, ty = x_1, y_1
    x_1, y_1, mse = brute_search(prev_frame_gray,frame_gray,x_1,y_1,4,16)
    mse_list.append(mse)
    print(f'At frame {cnt+1}. MV is ({x_1 - tx}, {y_1 - ty}) and the MSE is {mse}')
    cv2.rectangle(frame, (x_1, y_1), (x_1+16, y_1+16), (0,0,0))
    cv2.imshow("capture", frame)
    cv2.waitKey()
    prev_frame = frame
plt.figure()
plt.plot([f for f in range(1, frame_num+1)], mse_list)
plt.show()
