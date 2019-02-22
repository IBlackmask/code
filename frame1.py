import numpy as np
import cv2
import math
import os

from networktables import NetworkTables

NetworkTables.initialize(server='roborio-7086-frc.local')

sd = NetworkTables.getTable('Vision')
font = cv2.FONT_HERSHEY_COMPLEX
CONTOUR_LIMIT = 1


def onemli_bolgeler(img, vertices):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_frame = cv2.bitwise_and(img, mask)
    return masked_frame


capture = cv2.VideoCapture(1)

while (capture.isOpened()):
    ret, frame = capture.read()
    if ret:
        ##img_resized = cv2.resize(frame, (0, 0), fx=0.320, fy=0.240)
        img_resized = cv2.medianBlur(frame, 3)
        img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
        lower_white = np.array([0, 0, 200], dtype=np.uint8)
        upper_white = np.array([180, 150, 255], dtype=np.uint8)
        mask = cv2.inRange(img_hsv, lower_white, upper_white)
        res = cv2.bitwise_and(img_resized, img_resized, mask=mask)
        canny = cv2.Canny(mask, 200, 250)
        imshape = img_resized.shape
        lower_left = [imshape[1] / 9, imshape[0]]
        lower_right = [imshape[1] - imshape[1] / 10, imshape[0]]
        top_left = [imshape[1] / 2 - imshape[1] / 2, imshape[0] / 2 + imshape[0] / 9]
        top_right = [imshape[1] / 2 + imshape[1] / 2, imshape[0] / 2 + imshape[0] / 9]
        vertices = [np.array([lower_left, top_left, top_right, lower_right], dtype=np.int32)]
        roi_frame = onemli_bolgeler(canny, vertices)
        dary, contours, _ = cv2.findContours(roi_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            contourArea = cv2.contourArea(contour)
            if contourArea < 1:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            if w > 1 and w < 50:
                continue

            centerX = x + (w / 2)
            centerY = y + (h / 2)
            ratio = float(w) / h
            outImage = cv2.drawContours(img_resized, contour, -1, (255, 0, 0), 2)
            ##cv2.putText(img_resized, 'Merkez X Noktasi: :', (5, 70), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            print("Merkez X: ", centerX)
            print("Merkez Y: ", centerY)
            sd.putNumber('X', centerX)
            sd.putNumber('Y', centerY)

        cv2.imshow('DeepSpace', img_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        cv2.destroyAllWindows()
        break




