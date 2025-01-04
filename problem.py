import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from test import extract_sudoku

model = tf.keras.models.load_model('se_cnn_mnist_28x28.h5')

def remove_grid_from_sudoku(image):
    """
    Xóa các đường kẻ lưới trong ảnh Sudoku và giữ lại các số.
    """
    # Chuyển ảnh sang xám (grayscale)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Áp dụng threshold để chuyển ảnh thành nhị phân (để dễ nhận diện các đường kẻ)
    _, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Tạo kernel cho phép toán hình thái học
    kernel = np.ones((3, 3), np.uint8)
    
    # Áp dụng phép toán dilate để kết hợp các đường kẻ dọc và ngang
    dilated_image = cv2.dilate(binary_image, kernel, iterations=2)
    cv2.imshow("dilate",dilated_image)
    # Tìm các đường thẳng dọc và ngang bằng Hough Transform
    lines = cv2.HoughLinesP(dilated_image, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    
    # Tạo ảnh trống để vẽ lại các đường kẻ
    grid_image = np.zeros_like(image, dtype=np.uint8)  # Đảm bảo rằng grid_image có kiểu uint8
    
    # Vẽ các đường kẻ dọc và ngang tìm được từ Hough Transform
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(grid_image, (x1, y1), (x2, y2), (255, 255, 255), 2)  # Vẽ đường kẻ màu trắng
    
    # Chuyển grid_image thành ảnh nhị phân (đảm bảo kiểu dữ liệu là uint8)
    grid_image_gray = cv2.cvtColor(grid_image, cv2.COLOR_BGR2GRAY)
    _, grid_mask = cv2.threshold(grid_image_gray, 1, 255, cv2.THRESH_BINARY)
    # Chuyển ảnh sang xám (grayscale)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Áp dụng threshold để chuyển ảnh thành nhị phân (để dễ nhận diện các đường kẻ)
    _, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Sử dụng bitwise để xóa các đường kẻ khỏi ảnh gốc
    result_image = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(grid_mask))
    
    # Hiển thị ảnh đã xử lý
    cv2.imshow("Result Image without Grid", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return result_image

def remove_border_and_center_number(image, border_width=5, min_dimension_percent=0.05):
    """
    Hàm cắt bỏ viền và căn giữa chữ số vào nền đen.

    Args:
        image: Ảnh đầu vào (có thể chứa chữ số).
        target_size: Kích thước ảnh đầu ra (mặc định 32x32).
        border_width: Độ rộng viền cần bỏ (mặc định là 5).

    Returns:
        Hình ảnh đã được căn giữa vào nền đen với kích thước target_size.
    """
    # Lấy kích thước ảnh
    h, w = image.shape

    # Tạo ảnh nhị phân để tìm vùng có chữ số
    _, binary = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)    # Tìm contours trên ảnh nhị phân
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours :
        return image
    # Tính min_dimension dựa trên phần trăm kích thước ảnh
    img_height, img_width = image.shape
    min_dimension = min(img_width, img_height) * min_dimension_percent

    # Tìm bounding box hợp lệ
    valid_box = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= min_dimension and h >= min_dimension:  # Chỉ kiểm tra chiều rộng và chiều cao
            valid_box = (x, y, w, h)
            break

    # Nếu không tìm thấy bounding box hợp lệ, trả về ảnh full đen
    if not valid_box:
        return np.zeros(image.shape, dtype=np.uint8)

    x, y, w, h = valid_box
    # Cắt số từ ảnh
    number_crop = image[y:y+h, x:x+w]

    # Tạo nền đen mới với kích thước target_size
    final_image = np.zeros((image.shape), dtype=np.uint8)

    # Tính toán tọa độ để đặt số vào giữa nền đen
    center_x = (final_image.shape[1] - w) // 2
    center_y = (final_image.shape[0] - h) // 2

    # Đặt số vào giữa nền đen
    final_image[center_y:center_y+h, center_x:center_x+w] = number_crop
    # if np.any(final_image == 255) == False:
    # # Trả về ảnh full đen nếu không có pixel trắng
    #     return np.zeros(target_size, dtype=np.uint8)
    # Đảm bảo ảnh có kích thước target_size (nếu cần resize thêm)
    result = cv2.resize(final_image, image.shape)

    return result

def divide_and_threshold(img, save_path=None):
    """
    
    Chia ảnh thành 81 ô và áp dụng threshold trên mỗi ô.
    
    Args:
        img: Ảnh đã được làm thẳng (đã cắt và xử lý).
        save_path: Thư mục để lưu từng ô (mặc định là không lưu).
    
    Returns:
        List chứa 81 ảnh tương ứng với các ô Sudoku.
    """
    # Kiểm tra thư mục lưu nếu cần
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    # Lấy kích thước ảnh
    height, width = img.shape[:2]
    cell_height = height // 9
    cell_width = width // 9

    # Danh sách chứa các ô
    cells = []

    for row in range(9):
        for col in range(9):
            # Cắt từng ô
            x_start = col * cell_width
            x_end = (col + 1) * cell_width
            y_start = row * cell_height
            y_end = (row + 1) * cell_height
            cell = img[y_start:y_end, x_start:x_end]

            # Áp dụng threshold
            # _, cell_thresh = cv2.threshold(cell, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            
            new_cell_thresh = remove_border_and_center_number(cell)
            # print(np.sum(new_cell_thresh))
            # if  np.sum(cell)<150000:
            #     sharpened_image = sharpen_image(new_cell_thresh)
            #     # Tạo kernel cho phép toán hình thái học
            #     kernel = np.ones((3, 3), np.uint8)
        
            #     # Áp dụng phép toán dilate để kết hợp các đường kẻ dọc và ngang
            #     new_cell_thresh = cv2.dilate(sharpened_image, kernel, iterations=1)
            # Lưu ô vào danh sách
            # Hiển thị ảnh sau khi xử lý
            # plt.imshow(new_cell_thresh, cmap='gray')
            # plt.title(f"Cell {row}, {col}")
            # plt.show()
            cells.append(cv2.resize(new_cell_thresh,(28,28)))

            # Lưu ảnh nếu save_path được cung cấp
            if save_path:
                cell_filename = f"{save_path}/cell_{row}_{col}.png"
                cv2.imwrite(cell_filename, new_cell_thresh)

    return cells

def distinguish_5_and_6(image):
    """
    Phân biệt giữa số 5 và số 6 dựa trên đặc trưng hình dạng mà không dùng tỷ lệ chiều rộng/chiều cao.
    
    Args:
        image: Ảnh chứa một ô chữ số (28x28).
    
    Returns:
        "5" hoặc "6" dựa trên phân tích hình dạng.
    """
    # Chuyển ảnh sang nhị phân
    _, binary = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
    
    # Tìm contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None  # Không có gì để kiểm tra

    # Chọn contour lớn nhất
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Tính thuộc tính hình dạng
    area = cv2.contourArea(largest_contour)        # Diện tích vùng khép kín
    hull_area = cv2.contourArea(cv2.convexHull(largest_contour))  # Diện tích vỏ lồi
    solidity = area / hull_area if hull_area > 0 else 0  # Độ đặc của vùng

    # Quy tắc phân biệt cải thiện:
    if solidity > 0.70:  # Số 6 thường có độ đặc cao hơn, do vùng khép kín nhiều
        return 6

    return 5

#  Run with CNN
def predict_digits_from_folder(folder_path="output_cells", model=None):
    """Đọc ảnh từ thư mục, dự đoán số và trả về danh sách chữ số (0 cho ô trống)."""
    predicted_digits = []
    
    # Lấy danh sách các tệp trong thư mục, đảm bảo sắp xếp đúng thứ tự
    file_names = sorted(os.listdir(folder_path), key=lambda x: int(x.split('_')[1]) * 9 + int(x.split('_')[2].split('.')[0]))
    
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        
        # Đọc ảnh grayscale
        cell = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        backup = cell.copy()
        # Tính tỷ lệ pixel trắng trong ảnh
        white_pixels = np.sum(cell == 255)  # Số pixel trắng
        total_pixels = cell.size
        white_ratio = white_pixels / total_pixels  # Tỷ lệ pixel trắng
        
        # Xác định vùng trung tâm của ô (70% kích thước của ảnh)
        center_x, center_y = cell.shape[1] // 2, cell.shape[0] // 2
        center_width, center_height = int(cell.shape[1] * 0.7), int(cell.shape[0] * 0.7)  # Vùng trung tâm 70%
        center_area = cell[center_y - center_height // 2: center_y + center_height // 2, 
                           center_x - center_width // 2: center_x + center_width // 2]
        
        # Kiểm tra sự hiện diện của pixel đen trong vùng trung tâm
        black_pixels_in_center = np.sum(center_area > 0)  # Số pixel không phải trắng
        
        if black_pixels_in_center == 0:  # Nếu không có pixel đen
            predicted_digits.append(0)  # Gán ô này là trống (0)
        else:
            
            cell = cv2.resize(cell, (28, 28))
            # Chuẩn bị dữ liệu cho CNN
            cell = cell.astype('float32') / 255.0  # Chuẩn hóa
            cell = cell.reshape(1, 28, 28, 1)  # Thêm batch dimension
            
            # Dự đoán
            prediction = model.predict(cell)
            predicted_digit = np.argmax(prediction, axis=1)[0]
            # Thêm số dự đoán vào danh sách
            
            if(predicted_digit ==5 or predicted_digit==6):
                predicted_digit= distinguish_5_and_6(backup)
            predicted_digits.append(predicted_digit)

    return predicted_digits

def create_sudoku_matrix_from_predictions(predicted_digits):
    """Chuyển danh sách số thành ma trận 9x9."""
    matrix = np.array(predicted_digits).reshape(9, 9)
    return matrix

def cells_to_sudoku_matrix(predicted_digits):
    # Chuyển danh sách các chữ số dự đoán thành ma trận 9x9
    matrix = np.zeros((9, 9), dtype=int)
    for i, digit in enumerate(predicted_digits):
        row, col = divmod(i, 9)  # Tính hàng và cột từ chỉ số
        matrix[row][col] = digit
    return matrix

def is_valid_move(board, row, col, num):
    """Kiểm tra xem số 'num' có hợp lệ tại vị trí (row, col) không."""
    # Kiểm tra hàng
    if num in board[row]:
        return False
    
    # Kiểm tra cột
    if num in [board[i][col] for i in range(9)]:
        return False
    
    # Kiểm tra trong ô 3x3
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(start_row, start_row + 3):
        for j in range(start_col, start_col + 3):
            if board[i][j] == num:
                return False
    
    return True

def solve_sudoku(board):
    """Giải Sudoku sử dụng thuật toán Backtracking."""
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:  # Ô trống
                for num in range(1, 10):  # Thử các số từ 1 đến 9
                    if is_valid_move(board, row, col, num):
                        board[row][col] = num
                        
                        if solve_sudoku(board):  # Đệ quy
                            return True
                        
                        # Backtrack
                        board[row][col] = 0
                
                return False  # Không tìm được số hợp lệ
    return True

def draw_sudoku(board):
    """Hiển thị ma trận Sudoku ở định dạng dễ đọc."""
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("-" * 21)  # Đường kẻ ngang giữa các khối 3x3
        
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print("|", end=" ")  # Đường kẻ dọc giữa các khối 3x3
            
            print(board[i][j] if board[i][j] != 0 else ".", end=" ")
        print()  # Xuống dòng sau mỗi hàng

def save_image(image, image_name, output_folder='processed_images'):
    """Save the image to the output folder with the specified name."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Create the output folder if it doesn't exist
    file_path = os.path.join(output_folder, image_name)
    cv2.imwrite(file_path, image)

def main(file):
    extract_sudoku(file)
    img=remove_grid_from_sudoku(cv2.imread('cropped_image.jpg'))
    divide_and_threshold(img, 'output_cells')
    # Thêm số dự đoán vào danh sách
# Dự đoán số từ các ảnh trong thư mục
    model = tf.keras.models.load_model('se_cnn_mnist_28x28.h5')  # Tải model CNN
    predicted_digits = predict_digits_from_folder('output_cells', model)
    
    # Tạo ma trận Sudoku từ kết quả dự đoán
    sudoku_matrix = create_sudoku_matrix_from_predictions(predicted_digits)
    print("Initial Sudoku matrix:")
    draw_sudoku(sudoku_matrix)
    
    # Giải Sudoku
    if solve_sudoku(sudoku_matrix):
        print("\nSolved Sudoku matrix:")
        draw_sudoku(sudoku_matrix)
    else:
        print("No solution exists for the given Sudoku.")


if __name__ == "__main__":
    main("6.jpg")
