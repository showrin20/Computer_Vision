import cv2 as cv


capture = cv.VideoCapture('video/media.mp4')
while True:
    isTrue, frame = capture.read()
    cv.imshow('frame', frame)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break
capture.release()
cv.destroyAllWindows()

