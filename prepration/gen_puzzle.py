import cv2 as cv
import numpy as np

if __name__=="__main__":
    img = cv.imread('./data/test.jpg')
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp = sift.detect(gray,None)
    img=cv.drawKeypoints(gray,kp,img)
    cv.imwrite('sift_keypoints.jpg',img)

