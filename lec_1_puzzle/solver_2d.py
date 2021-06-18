import os, sys
from sklearn.metrics import mean_squared_error
import cv2 as cv
import numpy as np
from time import sleep


def check_sims_T(mImg1, mImg2):

    oPiv = cv.cvtColor(mImg1, cv.COLOR_BGR2GRAY)
    oTar = cv.cvtColor(mImg2, cv.COLOR_BGR2GRAY)

    oPiv = oPiv[:1, :]
    oTar = oTar[-1:, :]

    return mean_squared_error(oPiv, oTar)

def check_sims_B(mImg1, mImg2):

    oPiv = cv.cvtColor(mImg1, cv.COLOR_BGR2GRAY)
    oTar = cv.cvtColor(mImg2, cv.COLOR_BGR2GRAY)

    oPiv = oPiv[-1:, :]
    oTar = oTar[:1, :]

    return mean_squared_error(oPiv, oTar)


def check_sims_L(mImg1, mImg2):

    oPiv = cv.cvtColor(mImg1, cv.COLOR_BGR2GRAY)
    oTar = cv.cvtColor(mImg2, cv.COLOR_BGR2GRAY)

    oPiv = oPiv[:, :1]
    oTar = oTar[:, -1:]

    return mean_squared_error(oPiv, oTar)

def check_sims_R(mImg1, mImg2):
    oPiv = cv.cvtColor(mImg1, cv.COLOR_BGR2GRAY)
    oTar = cv.cvtColor(mImg2, cv.COLOR_BGR2GRAY)

    oPiv = oPiv[:, -1:]
    oTar = oTar[:, :1]

    return mean_squared_error(oPiv, oTar)


def checker_tl_validity(mTop, mLeft, mWidth):
    oT_idx = np.argsort(mTop)[::-1]
    oL_idx = np.argsort(mLeft)[::-1]

    oIdx_sum = list()
    for t, l in zip(oT_idx, oL_idx):
        if (t<=mWidth) & (l<=mWidth):
            oIdx_sum.append(int(t+l))

    return oIdx_sum

def checker_top_left(mData, mWidth):
    oCheck_vals = np.zeros((len(mData), len(mData)))

    for i in range(oCheck_vals.shape[0]):
        for j in range(oCheck_vals.shape[0]):
            if (i!=j):
                oCheck_vals[i][j] = check_sims_T(mData[i], mData[j])
            else:
                oCheck_vals[i][j] = np.inf

    oMins_T = np.amin(oCheck_vals, axis=0)

    oCheck_vals = np.zeros((len(mData), len(mData)))
    for i in range(oCheck_vals.shape[0]):
        for j in range(oCheck_vals.shape[0]):
            if (i!=j):
                oCheck_vals[i][j] = check_sims_L(mData[i], mData[j])
            else:
                oCheck_vals[i][j] = np.inf

    oMins_L = np.amin(oCheck_vals, axis=1)
    oVal = checker_tl_validity(oMins_T, oMins_L, mWidth=mWidth)

    return np.argmin(oVal)


def solver_2d(mData, mTop_left_idx, mWidth):
    oSeq = [mTop_left_idx]

    for i in range(mWidth-len(oSeq)):
        oMin = 9999999
        oApp_idx = -1

        for j in range(len(mData)):
            if not j in oSeq:
                oNew_min = check_sims_R(mData[oSeq[-1]], mData[j])
                if oNew_min < oMin:
                    oMin = oNew_min
                    oApp_idx = j

        oSeq.append(oApp_idx)

    #oImg_out = mData[oSeq[0]]
    #for seq in oSeq[1:]:
    #    oImg_out = np.concatenate((oImg_out, mData[seq]), axis=1)
    #return oImg_out
    return oSeq


def solver_1d(mData, mTop_idx, mWidth):
    oSeq = [mTop_idx]

    for i in range(mWidth-len(oSeq)):
        oMin = 9999999
        oApp_idx = -1

        for j in range(len(mData)):
            if not j in oSeq:
                oNew_min = check_sims_B(mData[oSeq[-1]], mData[j])
                if oNew_min < oMin:
                    oMin = oNew_min
                    oApp_idx = j

        oSeq.append(oApp_idx)

    oImg_out = mData[oSeq[0]]

    for seq in oSeq[1:]:
        oImg_out = np.concatenate((oImg_out, mData[seq]), axis=0)

    return oImg_out


if __name__=="__main__":
    filepath = '../lec_0_prepration/data/output/2d/'
    files = os.listdir(filepath)
    files.sort()
    print(files)

    img_lst = list()
    for afile in files:
        img_lst.append(cv.imread(filepath + afile))

    top_left_idx = checker_top_left(img_lst, mWidth=8)
    first_lines = solver_2d(img_lst, top_left_idx, mWidth=8)

    print(check_sims_B(img_lst[40], img_lst[48]))
    print(check_sims_B(img_lst[40], img_lst[48]))
    exit()

    for i in range(8):
        img_res = solver_1d(img_lst, first_lines[i], mWidth=8)
        #exit()
        #img_res = solver_2d(img_lst, top_left_idx, mWidth=8)
        cv.imshow('result', img_res)
        cv.waitKey(0)
    exit()



