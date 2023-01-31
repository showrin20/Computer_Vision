import cv2 as cv
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation = cv.INTER_AREA)

capture = cv.VideoCapture('video/media.mp4')
while True:
    isTrue, frame = capture.read()
    frame_rescaled = rescale_frame(frame, percent=50)
    cv.imshow('frame', frame)
    cv.imshow(' frame_rescaled',  frame_rescaled)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break
def changeRes(width,height):
    capture.set(3,width)
    capture.set(4,height)





capture.release()
cv.destroyAllWindows()