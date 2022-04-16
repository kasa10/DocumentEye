import pytesseract
import cv2
import imutils
import os
import nltk as nl



image = cv2.imread('prikaz.jpg')
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

text = pytesseract.image_to_string(image, lang='rus')
print(text)


 # Создать папки
file_dir = "bd_txt/"
if not os.path.isdir(file_dir):
    os.makedirs(file_dir)

file_dir = "bd_img/"
if not os.path.isdir(file_dir):
    os.makedirs(file_dir)

file_dir = "dogovor/"
if not os.path.isdir(file_dir):
    os.makedirs(file_dir)

file_dir = "rasporyazhenie/"
if not os.path.isdir(file_dir):
    os.makedirs(file_dir)

file_dir = "buhgalter/"
if not os.path.isdir(file_dir):
    os.makedirs(file_dir)

file_dir = "kadri/"
if not os.path.isdir(file_dir):
    os.makedirs(file_dir)

file_dir = "drugoe/"
if not os.path.isdir(file_dir):
    os.makedirs(file_dir)


file_dir = "signature/"   #для подписей
if not os.path.isdir(file_dir):
    os.makedirs(file_dir)



i=1 #ТУТ можно цилом присваивать имена для скана и оцифровоного теккста к нему, пусть сейчас будет 1
cv2.imwrite(r"bd_img/"+str(i)+'.jpg', image)


f = open(r"bd_txt/"+str(i)+'.txt','w')  # запись теккста в файл с тем же именем
f.write(text)  # запись
f.close()  # закрытие файла

#ОПРЕДЕЛЯЕМ тип(КЛАССИФИЦИРУЕМ) в зависимости от КЛЮЧЕВОГО СЛОВА
keywords=['приказ','договор','накладная','счет-фактура','постановление','cлужебная записка']

text=text.lower() #текст переводим в нижний регистр

text_tokens = nl.word_tokenize(text) #ТОКЕНИЗИРУЕМ

#print(text_tokens)

print('                                       ')
k=[0,'документ']
try:
    try:
        for x in keywords:
            if text_tokens.index(x) > k[0]:
                k[0]=text_tokens.index(x)
                k[1]=x
    except:
        pass
        #print('Это не',x)

    print('Тип документа:',k[1])
except:
    print('Неопределенный тип документа')


#Определение наименования организации, идет до слова определяющего тип докуемента

organ=' '.join(text_tokens[:k[0]])
print('НАИМЕНОВАНИЕ ОРГАНИЗАЦИИ:',organ)


#Определение расшифровки подписи (автора документа)
surname=' '.join(text_tokens[-3:])
print('ФАМИЛИЯ АВТОРА: ',surname)

i=1 #ТУТ можно циклом присваивать имена, но пусть будет 1

f = open(r"signature/"+str(i)+'.txt','w')  # запись раасшифровки в файл с тем же именем
f.write(surname)  # запись
f.close()  # закрытие файла

#Определение должности автора
# position=' '.join(text_tokens[-5:-3]) #ПОСЛЕ ТОЧКИ ПЕРЕД НАЧАЛОМ НАИМЕНОВАНИЕ ДОЛЖНОСТИ, НО ДО ФАМИЛИИ
#print('ДОЛЖНОСТЬ АВТОРА: ',position)

#ЗАПИСЬ ДАННЫХ В БД!!!
#import pymysql
#import pymysql.cursors

# con = pymysql.connect(host='localhost', user='root',password='', db='documenteye', cursorclass=pymysql.cursors.DictCursor)
#
# with con:
#     cur = con.cursor()

#     cur.execute("INSERT INTO rasporyazhenie(`Наименование организации`,`Тип документа`,`Автор`) VALUES (%s, %s, %s)", (organ, k[1], surname)))
#     con.commit()
