import numpy as np  
import cv2 as cv  
im = cv.imread("E:\opencv\photos\red_ball.jpg")
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)  
ret, thresh = cv.threshold(imgray, 127, 255, 0)  
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
