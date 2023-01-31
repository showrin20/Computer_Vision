import cv2 as cv
img = cv.imread('photos/yy.jpg')
cv.imshow('yy' ,img)
cv.waitKey(0)
# Extracting the height and width of an image
h, w = img.shape[:2]
# Displaying the height and width
print(h,w)