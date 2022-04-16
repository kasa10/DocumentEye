import imutils
from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2
import os

 # Создать папки
file_dir = "signature/"
if not os.path.isdir(file_dir):
    os.makedirs(file_dir)


image = cv2.imread('Prikaz.jpg')
orig = image.copy()
(H, W) = image.shape[:2]

(newW, newH) = (1024, 1024)
rW = W / float(newW)
rH = H / float(newH)
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

print("EAST text detector...")
net = cv2.dnn.readNet('frozen_east_text_detection.pb')

blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)

start = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
end = time.time()
print("Text detection took {:.6f} seconds".format(end - start))



def decode_predictions(scores, geometry, probThr=0.8):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        # loop over the number of columns
        for x in range(0, numCols):
            if scoresData[x] < probThr:
                continue
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            endX = int(offsetX + (cos * xData1[x]) + (sin *
                                                      xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos *
                                                      xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
    return confidences, rects


def draw_boxes(image, boxes, ration):
    (rW, rH) = ration

    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        # draw the bounding box on the image
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255,
                                                              0), 2)

    image1 = imutils.resize(image, height=700)

    cv2.imshow("Detected texts", image1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



confidences, rects = decode_predictions(scores, geometry)
boxes = non_max_suppression(np.array(rects), probs=confidences)
draw_boxes(orig, boxes, (rW, rH))

#НЕРАСПОЗНАНЫЙ КУСОК ОБЛАСТЬ ПОДПИСИ, ВЫРЕЗАЕМ И СОХРАНЯЕМ этот кусок ДЛЯ ОБУЧЕНИЯ

t=0
# for i in range(0,len(boxes)-1):  #находим области расшифровки подписи и должности внизу страницы (по Y)
#     if boxes[i][1] < boxes[i+1][1]:
#         t = boxes[i]
#         boxes[i] = boxes[i+1]
#         boxes[i+1] = t

m_Y=[]
m_X=[]
m_X2=[]
for i in boxes:
    m_Y.append(i[1])

print(m_Y.index(max(m_Y)))
print(max(m_Y))

for i in boxes:
    if max(m_Y)-6 > i[1]: #было max(m_Y)-6
        m_X.append(i[2])
        m_X2.append(i[0])

print(m_X)


signature_startX = int((min(m_X)+250) * rW)
signature_startY = int((boxes[m_Y.index(max(m_Y))][1]-90) * rH)
signature_endX = int((max(m_X2)-10)  * rW)
signature_endY = int((boxes[m_Y.index(max(m_Y))][3]+60) * rH)


cv2.rectangle(image, (signature_startX, signature_startY), (signature_endX, signature_endY), (255, 0,
                                                              0), 2)
image1=imutils.resize(image, height=700)
# show the output image
cv2.imshow("Detected signature", image1)
cv2.waitKey(0)
cv2.destroyAllWindows()

#СОХРАНЯЕМ ПОДПИСЬ И РАСШИФРОВУ В ФАЙЛ signature/
print('                      ')
print('                      ')
print((signature_startX, signature_startY), (signature_endX, signature_endY))
print(image.shape[:2])

signature_jpg = image[signature_startY:signature_endY, signature_startX:signature_endX]
cv2.imshow("Signature", signature_jpg)
cv2.waitKey(0)

n=1 #ТУТ можно циклом, но пусть сейчас будет 1
cv2.imwrite(r"signature/"+str(n)+'.jpg', signature_jpg)

#ЗАПИСЬ РАСШИФРОВКИ ВВ ФАЙЛЕ tesseract1.py