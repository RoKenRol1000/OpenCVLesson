import cv2 as cv

def test(suptractor):
  capture = cv.VideoCapture(cv.samples.findFileOrKeep("./videoplayback (3).webm"))
  while True:
    ret, frame = capture.read()
    if frame is None:
        break

    fgMask = suptractor.apply(frame)

    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

    cv.imshow("Frame", frame)
    cv.imshow("Mask", fgMask)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

test(cv.createBackgroundSubtractorKNN())