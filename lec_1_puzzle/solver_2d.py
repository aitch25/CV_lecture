import os, sys
import cv2 as cv
import numpy as np

def solver(mData):
    

    return


if __name__=="__main__":
    filepath = '../prepration/data/output/1d/'
    files = os.listdir(filepath)

    img_lst = list()
    for afile in files:
        img_lst.append(cv.imread(filepath + afile))





