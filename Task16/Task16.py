import cv2 as cv


def test(subtractor, on_change):
    capture = cv.VideoCapture(cv.samples.findFileOrKeep("./videoplayback (3).webm"))
    window = cv.namedWindow("Frame")
    cv.createTrackbar('slider', "Frame", 0, 255, on_change)
    while True:
        ret, frame = capture.read()
        if frame is None:
           break

        # Блюр чтобы убрать помехи на изображении
        # frame = cv.blur(frame, (3, 3))
        fgMask = subtractor.apply(frame)

        cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))


        cv.imshow("Frame", frame)
        cv.imshow("Mask", fgMask)
        cv.imshow("MaskedFrame", cv.bitwise_and(frame, frame, mask=fgMask))

        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break

if __name__ == '__main__':
    # KNN
    subtractor = cv.createBackgroundSubtractorKNN(detectShadows=True, dist2Threshold=1)
    test(subtractor, on_change=lambda t: subtractor.setDist2Threshold(t))
    # MOG2
    # subtractor = cv.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=1)
    # test(suptractor, on_change=lambda t: suptractor.setVarThreshold(t))