import cv2 as cv
import os
import imutils

hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

cascPathFace = os.path.dirname(cv.__file__) + "/data/haarcascade_fullbody.xml"
humanClassifier = cv.CascadeClassifier(cascPathFace)


#TODO: Вынести параметры + Refactor
def detectHuman(frame):
    frame = imutils.resize(frame, width=min(frame.shape[1], 800))
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    (humans, _) = hog.detectMultiScale(frame,
                                    winStride=(4, 4), padding=(8, 8), scale=1.1)
    hogImage = frame.copy()
    for (x, y, w, h) in humans:
        cv.rectangle(hogImage, (x, y), (x + w, y + h), (0, 255, 0), 2)

    humans = humanClassifier.detectMultiScale(gray, scaleFactor=1.05,
                                                  minNeighbors=5,
                                                  flags=cv.CASCADE_SCALE_IMAGE)
    classifierImage = frame.copy()
    for (x, y, w, h) in humans:
        cv.rectangle(classifierImage, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.imshow('Classifier', classifierImage)
    cv.imshow('HOG', hogImage)


cv.startWindowThread()

video_capture = cv.VideoCapture("videoplayback.webm")
while video_capture.isOpened():
    ret, frame = video_capture.read()
    detectHuman(frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv.destroyAllWindows()