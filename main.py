import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from preprocess import extract_sudoku

model = tf.keras.models.load_model('se_cnn_mnist_28x28.h5')

def remove_grid_from_sudoku(image):
    """
    Description: This function removes the grid lines from the Sudoku puzzle image. It converts the image to grayscale, applies thresholding and dilation to detect grid lines, and then applies a mask to remove those lines.
    Parameters:
        image (numpy array): The input image containing the Sudoku puzzle with grid lines.
    Returns:
        result_image (numpy array): The image with the grid lines removed.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV, )

    kernel = np.ones((3, 3), np.uint8)

    dilated_image = cv2.dilate(binary_image, kernel, iterations=2)
    cv2.imshow("dilate",dilated_image)

    lines = cv2.HoughLinesP(dilated_image, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    grid_image = np.zeros_like(image, dtype=np.uint8) 

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(grid_image, (x1, y1), (x2, y2), (255, 255, 255), 2) 

    grid_image_gray = cv2.cvtColor(grid_image, cv2.COLOR_BGR2GRAY)

    _, grid_mask = cv2.threshold(grid_image_gray, 1, 255, cv2.THRESH_BINARY)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)

    result_image = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(grid_mask))

    cv2.imshow("Result Image without Grid", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result_image

def remove_border_and_center_number(image, border_width=5, min_dimension_percent=0.05):
    """
    Description: This function removes the border of the number in the Sudoku cell and centers the number within the cell. It extracts the bounding box of the number and resizes it to the center of a black image.
    Parameters:
        image (numpy array): The input image of a Sudoku cell.
        border_width (int, optional): Width of the border to remove (default is 5).
        min_dimension_percent (float, optional): Minimum dimension of the cell as a percentage of the total image size (default is 0.05).
    Returns:
        result (numpy array): The image of the centered number in the cell.
    """
    h, w = image.shape

    _, binary = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)  

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours :
        return image

    img_height, img_width = image.shape
    min_dimension = min(img_width, img_height) * min_dimension_percent

    valid_box = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= min_dimension and h >= min_dimension:  
            valid_box = (x, y, w, h)
            break


    if not valid_box:
        return np.zeros(image.shape, dtype=np.uint8)

    x, y, w, h = valid_box
    number_crop = image[y:y+h, x:x+w]

    final_image = np.zeros((image.shape), dtype=np.uint8)

    center_x = (final_image.shape[1] - w) // 2
    center_y = (final_image.shape[0] - h) // 2

    final_image[center_y:center_y+h, center_x:center_x+w] = number_crop

    result = cv2.resize(final_image, image.shape)

    return result

def divide_cell(img, save_path=None):
    """
    Description: This function divides a Sudoku image into individual 9x9 cells and applies the remove_border_and_center_number function to each cell. It also saves each individual cell image if a save path is provided.
    Parameters:
        img (numpy array): The input image of the Sudoku puzzle.
        save_path (str, optional): Path to save individual cell images (default is None).
    Returns:
        cells (list): A list of processed cell images.
    """
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    height, width = img.shape[:2]
    cell_height = height // 9
    cell_width = width // 9

    cells = []

    for row in range(9):
        for col in range(9):
            x_start = col * cell_width
            x_end = (col + 1) * cell_width
            y_start = row * cell_height
            y_end = (row + 1) * cell_height
            cell = img[y_start:y_end, x_start:x_end]

            new_cell_thresh = remove_border_and_center_number(cell)

            cells.append(cv2.resize(new_cell_thresh,(28,28)))

            if save_path:
                cell_filename = f"{save_path}/cell_{row}_{col}.png"
                cv2.imwrite(cell_filename, new_cell_thresh)

    return cells

def distinguish_5_and_6(image):
    """
    Description: This function distinguishes between the digits 5 and 6 in a Sudoku cell image by analyzing the solidity of the largest contour in the cell. A solidity greater than 0.7 indicates the digit is a 6.
    Parameters:
        image (numpy array): The input image of a Sudoku cell.
    Returns:
        5 or 6 (int): The predicted digit.
    """
    _, binary = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None  

    largest_contour = max(contours, key=cv2.contourArea)
    
    area = cv2.contourArea(largest_contour)     
    hull_area = cv2.contourArea(cv2.convexHull(largest_contour))  
    solidity = area / hull_area if hull_area > 0 else 0 

    if solidity > 0.70:  
        return 6

    return 5

def predict_digits_from_folder(folder_path="output_cells", model=None):
    """
    Description: This function predicts the digits of a Sudoku puzzle from a folder containing individual cell images. Each image is resized, normalized, and passed through the trained CNN model to predict the digit.
    Parameters:
        folder_path (str, optional): The folder containing the cell images (default is "output_cells").
        model (tf.keras.Model, optional): The trained model used for digit prediction.
    Returns:
        predicted_digits (list): A list of predicted digits for the 81 cells in the Sudoku puzzle.
    """
    predicted_digits = []
    
    file_names = sorted(os.listdir(folder_path), key=lambda x: int(x.split('_')[1]) * 9 + int(x.split('_')[2].split('.')[0]))
    
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        
        cell = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        backup = cell.copy()

        center_x, center_y = cell.shape[1] // 2, cell.shape[0] // 2
        center_width, center_height = int(cell.shape[1] * 0.7), int(cell.shape[0] * 0.7)  
        center_area = cell[center_y - center_height // 2: center_y + center_height // 2, 
                           center_x - center_width // 2: center_x + center_width // 2]
        
        black_pixels_in_center = np.sum(center_area > 0)  
        
        if black_pixels_in_center == 0: 
            predicted_digits.append(0)  
        else:
            
            cell = cv2.resize(cell, (28, 28))
            cell = cell.astype('float32') / 255.0  
            cell = cell.reshape(1, 28, 28, 1)  
            
            prediction = model.predict(cell)
            predicted_digit = np.argmax(prediction, axis=1)[0]
            
            if(predicted_digit ==5 or predicted_digit==6):
                predicted_digit= distinguish_5_and_6(backup)
            predicted_digits.append(predicted_digit)

    return predicted_digits

def create_sudoku_matrix_from_predictions(predicted_digits):
    """
    Description: This function creates a Sudoku matrix from the list of predicted digits.
    Parameters:
        predicted_digits (list): A list of predicted digits for the 81 cells in the Sudoku puzzle.
    Returns:
        matrix: A Sudoku matrix
    """
    matrix = np.array(predicted_digits).reshape(9, 9)
    return matrix

def cells_to_sudoku_matrix(predicted_digits):
    """
    Description: This function converts a list of predicted digits into a Sudoku matrix using the solve_sudoku function.
    Parameters:
        predicted_digits (list): A list of predicted digits for the 81 cells in the Sudoku puzzle.
    Returns:
        matrix: A Sudoku matrix
    """
    matrix = np.zeros((9, 9), dtype=int)
    for i, digit in enumerate(predicted_digits):
        row, col = divmod(i, 9)  
        matrix[row][col] = digit
    return matrix

def is_valid_move(board, row, col, num):
    """
    Description: This function checks if a given number can be placed in the specified cell without violating the Sudoku rules.
    Parameters:
        board (numpy array): The current Sudoku board.
        row (int): The row index of the cell.
    Returns:
        bool: True if the number can be placed in the specified cell without violating the Sudoku rules, False otherwise.
    """
    if num in board[row]:
        return False
    
    if num in [board[i][col] for i in range(9)]:
        return False
    
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(start_row, start_row + 3):
        for j in range(start_col, start_col + 3):
            if board[i][j] == num:
                return False
            
    return True

def solve_sudoku(board):
    """
    Description: This function solves the Sudoku puzzle using backtracking.
    Parameters:
        board (numpy array): The current Sudoku board.
    Returns:
        bool: True if the Sudoku puzzle is solved, False otherwise.
    """
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:  
                for num in range(1, 10):  
                    if is_valid_move(board, row, col, num):
                        board[row][col] = num

                        if solve_sudoku(board):  
                            return True

                        board[row][col] = 0

                return False  
    return True

def draw_sudoku(board):
    """
    Description: This function draws the Sudoku board using ASCII characters.
    Parameters:
        board (numpy array): The current Sudoku board.
    Returns:
        None: This function prints the Sudoku board to the console.
    """
    
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("-" * 21)  
        
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print("|", end=" ")  
            
            print(board[i][j] if board[i][j] != 0 else ".", end=" ")
        print()  

def save_image(image, image_name, output_folder='processed_images'):
    """
    Description: This function saves the input image to the specified output folder.
    Parameters:
        image (numpy array): The input image to be saved.
        image_name (str): The name of the image file to be saved
    Returns:
        None: This function saves the input image to the specified output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  
    file_path = os.path.join(output_folder, image_name)
    cv2.imwrite(file_path, image)

def main(file):
    extract_sudoku(file)
    img=remove_grid_from_sudoku(cv2.imread('cropped_image.jpg'))
    divide_cell(img, 'output_cells')
    model = tf.keras.models.load_model('se_cnn_mnist_28x28.h5')  
    predicted_digits = predict_digits_from_folder('output_cells', model)
    sudoku_matrix = create_sudoku_matrix_from_predictions(predicted_digits)
    print("Initial Sudoku matrix:")
    draw_sudoku(sudoku_matrix)
    
    if solve_sudoku(sudoku_matrix):
        print("\nSolved Sudoku matrix:")
        draw_sudoku(sudoku_matrix)
    else:
        print("No solution exists for the given Sudoku.")

if __name__ == "__main__":
    main("input_images/1.jpg")
