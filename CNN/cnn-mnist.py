import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

# 数据输入的参数  
img_rows, img_cols = 28, 28
num_classes = 10

from keras.datasets.fashion_mnist import load_data
# # 加载训练集的图像及标签
(img_our, label_our), (_, _) = load_data()

print("原始图片规模：", img_our.shape,label_our.shape)

dict_labels_mnistfashion = {0: 'T-shirt', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt',
                            7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}
dict_labels_mnist = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

for c in range(1, 10):  # 创建了一个类别的不平衡
    img_our_B = np.vstack(
        [img_our[label_our != c], img_our[label_our == c][:100 * c]])
    label_our_B = np.append(label_our[label_our != c], np.ones(100 * c) * c)

(_, _), (testX, testy) = load_data()
for c in range(1, 10):
    img_our_B_test = np.vstack(
        [testX[testy != c], testX[testy == c][:7 * c]])

    label_our_B_test = np.append(testy[testy != c], np.ones(7 * c) * c)
    # 不平衡数据集设定
x_train_path = img_our_B
y_train_path = label_our_B
x_test_path = img_our_B_test
y_test_path = label_our_B_test

# 加载 MNIST 数据集  
(x_train, y_train), (x_test, y_test) = (x_train_path, y_train_path), (x_test_path, y_test_path)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# 对像素数据进行归一化  
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# 对标签进行 One-hot 编码  
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

# 构建模型  
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# 编译模型  
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

# 训练模型  
model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])