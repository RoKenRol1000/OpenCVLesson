import cv2
import os
cascPathFace = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPathFace)


def detectFace(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('FaceDetection', frame)

#для вебки
# video_capture = cv2.VideoCapture(0)
# while True:
#     ret, frame = video_capture.read()
#     detectFace(frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# video_capture.release()
# cv2.destroyAllWindows()

#Для картинки
image = cv2.imread("Face.jpg")
detectFace(image)
cv2.waitKey(0)