from imutils.perspective import four_point_transform
from skimage.filters import threshold_local
import cv2
import imutils
import os



 # Создать папку визуализации
file_dir = "vis/"
if not os.path.isdir(file_dir):
    os.makedirs(file_dir)

 # Читать картинку

image_path = "test.jpg"
image = cv2.imread(image_path)
ratio = image.shape[0] / 500.0
orig = image.copy()

 # Вырезать ввод
image = imutils.resize(image, height = 500)

 # Изображение в оттенках серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 # Выполнить обработку фильтра Гаусса
gray = cv2.GaussianBlur(gray, (5, 5), 0)
 # Выполнить обработку обнаружения края
edged = cv2.Canny(gray, 75, 200)

 # Отображение и сохранение результатов
print("1")
cv2.imshow("1", image)
cv2.imshow("2", edged)
cv2.imwrite("vis\edged.png", edged)

 # Найдите контуры на краях изображений и фильтруйте контуры с меньшими точками
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
 # Сортировка по размеру области и получение 5 лучших результатов
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

 # Обойти всю коллекцию контуров
for c in cnts:
	 # Используйте многоугольники для аппроксимации контуров
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	if len(approx) == 4:
		screenCnt = approx
		break

 # Отображение и сохранение результатов
print("2")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Y", image)
cv2.imwrite("vis\contours.png", image)

 # Использование координатных точек для преобразования координат
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

 # Преобразуем преобразованный результат в значение серого
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
 # Получить порог локальной территории
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
 # Выполнить бинаризационную обработку
warped = (warped > T).astype("uint8") * 255

 # Отображение и сохранение результатов
print("3")
cv2.imshow("ОРИГ", imutils.resize(orig, height = 650))
cv2.imshow("СКАН", imutils.resize(warped, height = 650))
cv2.imwrite("vis\orig.png", orig)
cv2.imwrite("vis\warped.png", warped)
cv2.waitKey(0)