import cv2
import numpy as np

def analyze_image(image):
    """
    Phân tích hình ảnh, in ra các thông số liên quan đến contours và hình dạng.
    
    Args:
        image: Ảnh chứa một ô chữ số (28x28).
    
    Returns:
        None
    """
    # Chuyển ảnh sang nhị phân
    _, binary = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
    
    # Tìm contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Không tìm thấy contours!")
        return
    
    print(f"Tổng số contours tìm thấy: {len(contours)}")
    
    for idx, contour in enumerate(contours):
        # Tính thuộc tính hình dạng
        area = cv2.contourArea(contour)  # Diện tích vùng khép kín
        hull_area = cv2.contourArea(cv2.convexHull(contour))  # Diện tích vỏ lồi
        solidity = area / hull_area if hull_area > 0 else 0  # Độ đặc của vùng
        
        # In các thông số cho từng contour
        print(f"\nContour {idx + 1}:")
        print(f"  - Area: {area}")
        print(f"  - Hull Area: {hull_area}")
        print(f"  - Solidity: {solidity}")
        
        # Vẽ contour lên ảnh để xem trực quan
        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(color_image, [contour], -1, (0, 255, 0), 2)
    
    # Hiển thị ảnh có các contours vẽ lên
    cv2.imshow("Contours", color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Đọc ảnh và gọi hàm phân tích
# image = cv2.imread('output_cells\cell_4_4.png', cv2.IMREAD_GRAYSCALE)  # Đảm bảo sử dụng đúng đường dẫn ảnh
image = cv2.imread('output_cells\cell_3_1.png', cv2.IMREAD_GRAYSCALE)  # Đảm bảo sử dụng đúng đường dẫn ảnh

analyze_image(image)
