import cv2 as cv
import numpy as np
blank= np.zeros((500,500,3), dtype='uint8')
cv.imshow('blank', blank)

#paint the image
blank[200:300,300:400]= 0,0,255
cv.imshow('blank', blank)

#draw a reactangle
#cv.rectangle(blank,(0,0), (blank.shape[1]//2, blank.shape[0]//2) ,(0,255,0),thickness= -1)
#cv.imshow('blank', blank)
#draw a cicle
#cv.circle(blank,(0,0), (blank.shape[1]//2, blank.shape[0]//2) , 40, (0,0,255), -1)
#cv.imshow('Circle', blank)
#cv.waitKey(10000)
cv.rectangle(blank,(0,0), (blank.shape[1]//2, blank.shape[0]//2) ,(0,255,0),thickness= -1)
cv.imshow('blank', blank)

cv.circle(blank,(0,0), (blank.shape[1]//2, blank.shape[0]//2) , 40, (0,0,255), -1)

cv.imshow('Circle', blank)

cv.waitKey(10000) 