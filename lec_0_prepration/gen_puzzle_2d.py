import sys
import cv2 as cv
import numpy as np
from time import sleep
from random import shuffle

if __name__=="__main__":
    p_cnt = int(sys.argv[1])

    img = cv.imread('./data/input/fruits.jpg')
    img = img[-1280:, :1280]

    new_size = 640
    img = cv.resize(img, dsize=(new_size, new_size))

    p_size = int(new_size / p_cnt)

    suf_idx = list(range(p_cnt * p_cnt))
    shuffle(suf_idx)

    idx = 0
    for i in range(p_cnt):
        for j in range(p_cnt):
            p_img = img[(p_size*i):(p_size*(i+1)), (p_size*j):(p_size*(j+1))]
            cv.imwrite('./data/output/2d/part_' + str(i) + str(j) + '.png', p_img)
            #cv.imwrite('./data/output/2d/part_' + str(suf_idx[idx]) + '.png', p_img)
            idx += 1

