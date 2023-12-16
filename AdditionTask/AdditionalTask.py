import cv2
import os
cascPathFace = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPathFace)

ageModels = './deploy_age.prototxt'
ageProto = './age_net.caffemodel'

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_INTERVALS = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
                 '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

age_net = cv2.dnn.readNetFromCaffe(ageModels, ageProto)



def detectFace(img):
    frame = img.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for (x, y, w, h) in faces:
        face_img = img.copy()[y: y+h, x: x+w]
        blob = cv2.dnn.blobFromImage(
            image=face_img, scalefactor=1.0, size=(227, 227),
            mean=MODEL_MEAN_VALUES, swapRB=False)

        age_net.setInput(blob)
        age_preds = age_net.forward()

        print("=" * 30, f"Face {i + 1} Prediction Probabilities", "=" * 30)
        for i in range(age_preds[0].shape[0]):
            print(f"{AGE_INTERVALS[i]}: {age_preds[0, i] * 100:.2f}%")
        i = age_preds[0].argmax()
        age = AGE_INTERVALS[i]
        age_confidence_score = age_preds[0][i]

        label = f"Age:{age} - {age_confidence_score * 100:.2f}%"
        print(label)

        yPos = y - 15
        while yPos < 15:
            yPos += 15

        cv2.putText(frame, label, (x, yPos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)

    cv2.imshow('FaceDetection', frame)

#Для картинки
image = cv2.imread("28-9.png")
detectFace(image)
cv2.waitKey(0)