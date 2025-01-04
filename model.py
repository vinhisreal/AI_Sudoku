import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as img
# Xây dựng mô hình CNN để phân loại từ 1 đến 9
def get_cnn_model():
    #CNN Architecture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> 
                           #Flatten -> Dense -> Dropout -> Out
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(filters=32, kernel_size=(5,5), padding='Same', 
                        activation=tf.nn.relu, input_shape = (28,28,1)))
    model.add(layers.Conv2D(filters=32, kernel_size=(5,5), padding='Same', 
                        activation=tf.nn.relu))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))


    model.add(layers.Conv2D(filters=64, kernel_size=(3,3), padding='Same', 
                        activation=tf.nn.relu, input_shape = (28,28,1)))
    model.add(layers.Conv2D(filters=64, kernel_size=(3,3), padding='Same', 
                        activation=tf.nn.relu))
    model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(256,activation=tf.nn.relu))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(10,activation=tf.nn.softmax))

# Huấn luyện và lưu mô hình CNN
def train_and_save_cnn_model():
    # Tải bộ dữ liệu MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Lọc dữ liệu để chỉ giữ lại các chữ số từ 1 đến 9
    x_train = x_train[y_train != 0]
    y_train = y_train[y_train != 0]
    x_test = x_test[y_test != 0]
    y_test = y_test[y_test != 0]

    # Tiền xử lý dữ liệu
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Chuyển đổi kích thước ảnh và chuẩn hóa ảnh (32x32)
    x_train_resized = [cv2.resize(img, (32, 32)) for img in x_train]
    x_test_resized = [cv2.resize(img, (32, 32)) for img in x_test]

    # Chuyển đổi thành mảng numpy
    x_train_resized = np.array(x_train_resized)
    x_test_resized = np.array(x_test_resized)

    # Chuyển ảnh thành dạng (batch_size, height, width, channels)
    x_train_resized = x_train_resized.reshape(-1, 32, 32, 1)
    x_test_resized = x_test_resized.reshape(-1, 32, 32, 1)

    # Chuyển nhãn thành one-hot encoding (9 lớp cho 1-9)
    y_train = to_categorical(y_train - 1, 9)  # Đảm bảo nhãn bắt đầu từ 0 (do MNIST có nhãn từ 1 đến 9)
    y_test = to_categorical(y_test - 1, 9)

    # Tạo mô hình CNN
    model = get_cnn_model()

    # Huấn luyện mô hình
    model.fit(x_train_resized, y_train, epochs=10, validation_data=(x_test_resized, y_test))

    # Lưu mô hình sau khi huấn luyện
    model.save('cnn_1_to_9.h5')  # Lưu mô hình vào file cnn_1_to_9.h5
    print("Mô hình CNN đã được lưu.")

# Chạy hàm huấn luyện và lưu mô hình CNN
if __name__ == "__main__":
    train_and_save_cnn_model()
   