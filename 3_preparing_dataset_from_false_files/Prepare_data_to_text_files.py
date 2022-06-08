# Подключаем необходимые библиотеки
import numpy as np
import cv2

# Зададим размер нашей выборки
COUNT_VAR_IMG = 2000 # т.е. всего у нас будет 42 тысячиизображений
IMG_HIGH = 32
IMG_WEIGTH = 32
IMG_COLOR_NUMBER = 3

# Вычислим длину для преобразования нашего тензора для помещения его в файл ".csv" (если будем использовать вообще)
Length_Tensor_4D_Flatten = COUNT_VAR_IMG * IMG_HIGH * IMG_WEIGTH * IMG_COLOR_NUMBER

# Сразу определяем, сколько у нас 0, 1
num_0 = 1000
num_1 = 1000

check_sum = num_0 + num_1

# Проверяем, чтобы все сошлось
if (check_sum != COUNT_VAR_IMG):
    print('check_sum is not same COUNT_VAR_IMG')
    exit(-1)


# Создаем наш 4D-массив. Назовем его Tensor_4D
Tensor_4D = np.zeros((COUNT_VAR_IMG, IMG_HIGH, IMG_WEIGTH, IMG_COLOR_NUMBER),)

print(Tensor_4D.shape)

# Создаем наш 2D-массив с метками классов (10 - это количество классов)
Labels = np.zeros((COUNT_VAR_IMG,),)
print(Labels.shape)


# Заполняем наш 4D массив (сразу дадим ему размер, необходимый для того, чтобы отдать его в CNN в Keras)
count = 0

# Для 0
for count_variable_image in range (0, num_0):
    imgorig = cv2.imread('gen_2/2_' + str(count_variable_image) + '.jpg', cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(imgorig, cv2.COLOR_BGR2RGB) # Это и есть одна выборка
    for i in range(rgb_img.shape[0]):
        for j in range(rgb_img.shape[1]):
            for c in range(rgb_img.shape[2]):
                # Это действие излишнее, т.к. rgb_image и так уже массив NumPy нужного размера
                Tensor_4D[count][i][j][c] = rgb_img[i][j][c]
    Labels[count] = 0
    count = count + 1


print('Обаработано изображений: ', count)


# Для 1
for count_variable_image in range (0, num_1):
    imgorig = cv2.imread('gen_3/3_' + str(count_variable_image) + '.jpg', cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(imgorig, cv2.COLOR_BGR2RGB) # Это и есть одна выборка
    for i in range(rgb_img.shape[0]):
        for j in range(rgb_img.shape[1]):
            for c in range(rgb_img.shape[2]):
                # Это действие излишнее, т.к. rgb_image и так уже массив NumPy нужного размера
                Tensor_4D[count][i][j][c] = rgb_img[i][j][c]
    Labels[count] = 1
    count = count + 1

print('Обаработано изображений: ', count)

print ('count_after == 2000 ? Answer: ', count)

# Проверка на правильность считывания
if (count < 2000 - 2):
    print('count != 2472 - 1')
    exit(-1)

# Выведем первые 20 пикселей 3-го изображения (3.jpg) из класса "2" для проверки правильности считывания (0 - с 0 по 4131, 1 - с 4132 по 8815, 2 - начинается с 8816 )
print(Tensor_4D[88][:][:][0])

# сохраним numpy-массив в файл формата .txt
f = open('data_set_real_data.txt', 'w')

for s in range(COUNT_VAR_IMG):
    for i in range(IMG_HIGH):
        for j in range(IMG_WEIGTH):
            for c in range(IMG_COLOR_NUMBER):
                # добавить в конец файла новую строку как отдельное значение массива values
                f.write(str(Tensor_4D[s][i][j][c]) + '\n')

f.close()

# сохраним метки в файл формата .txt
f = open('data_set_real_labels.txt', 'w')

for s in range(COUNT_VAR_IMG):
    f.write(str(Labels[s]) + '\n')

f.close()




print("Программа завершена успешно")


'''


# --------------------------------- Это часть уже должна будет распологаться в Jupyter notebook в  Google Colobaratory --------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

# Создаем наш Tensor-4D_new
Tensor_4D_new = np.zeros((COUNT_VAR_IMG, IMG_HIGH, IMG_WEIGTH, IMG_COLOR_NUMBER),)

# Считываем  из файла ".txt" наш тензор
f = open('data_set_real_data.txt', 'r')

for s in range(COUNT_VAR_IMG):
    for i in range(IMG_HIGH):
        for j in range(IMG_WEIGTH):
            for c in range(IMG_COLOR_NUMBER):
                Tensor_4D_new[s][i][j][c] = np.float(f.readline())

f.close()

# Создадим метки Labels_new
Labels_new = np.zeros((COUNT_VAR_IMG,),)


# Считаем метки из файл формата .txt
f = open('data_set_real_labels.txt', 'r')

for s in range(COUNT_VAR_IMG):
    Labels_new[s] = np.float(f.readline())

f.close()


# Приводим их к виду, которому они были сделаны изначально (пригодными для обучения в CNN библиотеки Keras)
print('Tensor_4D_new.shape = ', Tensor_4D_new.shape)
print('Labels_new.shape = ', Labels.shape)


# Проверяем на эквивалентность изначально считанным данным

# Проверка тензоров данных
equal = np.allclose(Tensor_4D_new,Tensor_4D)
print('Tensor_4D are equal: ',equal)


# Проверка меток классов
equal = np.allclose(Labels_new,Labels)
print('Labels are equal: ',equal)


# --------------------------------- Это часть уже должна будет распологаться в Jupyter notebook в  Google Colobaratory --------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

'''