import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import joblib  # Thư viện để lưu mô hình

model = tf.keras.models.load_model('model.h5')


# evaluate loaded model on test data
def identify_number(image):
    image_resize = cv2.resize(image, (28, 28)) 
    
    image_resize = image_resize.reshape(1, 28, 28, 1) 
    
    image_resize = image_resize.astype('float32') / 255.0
    
    prediction = model.predict(image_resize)
    
    return np.argmax(prediction)


def extract_number(sudoku):
    sudoku = cv2.resize(sudoku, (450,450))
#    cv2.imshow('sudoku', sudoku)

    # split sudoku
    grid = np.zeros([9,9])
    for i in range(9):
        for j in range(9):
#            image = sudoku[i*50+3:(i+1)*50-3,j*50+3:(j+1)*50-3]
            image = sudoku[i*50:(i+1)*50,j*50:(j+1)*50]
            filename = "images/file_%d_%d.jpg"%(i, j)
            cv2.imwrite(filename, image)
            if image.sum() > 80000:
                # if image.sum() > 160000:
                #     kernel = np.ones((3, 3), np.uint8)  # Tạo kernel
                #     image = cv2.erode(image, kernel, iterations=)
                grid[i][j] = identify_number(image)
            else:
                grid[i][j] = 0
    return grid.astype(int)





