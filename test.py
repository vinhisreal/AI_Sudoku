import numpy as np
import cv2
import operator
import numpy as np
from matplotlib import pyplot as plt
from readmodel import extract_number
import sys
import tensorflow as tf
import os

model = tf.keras.models.load_model('se_cnn_mnist_28x28.h5')

def show_image(img):
    """Shows an image until any key is pressed"""
    print(type(img))
    print(img.shape)
    cv2.imshow('image', img)  # Display the image
    cv2.imwrite('images/gau_sudoku3.jpg', img)
    cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
    cv2.destroyAllWindows()  # Close all windows
    return img


def show_digits(digits, model=model, colour=255):
    """Hiển thị danh sách 81 chữ số được trích xuất trong định dạng lưới và dự đoán."""
    rows = []
    predictions = []  # Lưu kết quả dự đoán

    # Thêm viền cho từng ảnh chữ số
    with_border = [
        cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, colour) for img in digits
    ]

    for i in range(9):
        row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=1)
        rows.append(row)

        # Dự đoán các chữ số trong hàng hiện tại
        for digit in digits[i * 9:((i + 1) * 9)]:
            # Chuẩn hóa đầu vào trước khi đưa vào mô hình (nếu cần)
            if digit is not None:
                digit = digit / 255.0  # Normalize pixel values to [0, 1]
                digit = digit.reshape(1, 28, 28, 1)  # Đảm bảo đầu vào phù hợp với mô hình
                pred = model.predict(digit)  # Dự đoán chữ số
                predictions.append(np.argmax(pred))  # Lưu kết quả dự đoán (chỉ số của giá trị cao nhất)
            else:
                predictions.append(0)  # Nếu không có chữ số, mặc định là 0

    # Hiển thị ảnh
    img = show_image(np.concatenate(rows))
    print("Predictions:", predictions)  # In kết quả dự đoán

    return img, predictions

 

def convert_when_colour(colour, img):
	"""Dynamically converts an image to colour if the input colour is a tuple and the image is grayscale."""
	if len(colour) == 3:
		if len(img.shape) == 2:
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		elif img.shape[2] == 1:
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	return img

def pre_process_image(img, skip_dilate=False):
	"""Uses a blurring function, adaptive thresholding and dilation to expose the main features of an image."""

	# Gaussian blur with a kernal size (height, width) of 9.
	# Note that kernal sizes must be positive and odd and the kernel must be square.
	proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)

	# Adaptive threshold using 11 nearest neighbour pixels
	proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
	# Invert colours, so gridlines have non-zero pixel values.
	# Necessary to dilate the image, otherwise will look like erosion instead.
	proc = cv2.bitwise_not(proc, proc)

	if not skip_dilate:
		# Dilate the image to increase the size of the grid lines.
		kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
		proc = cv2.dilate(proc, kernel)

	return proc

def find_corners_of_largest_polygon(img):
	"""Finds the 4 extreme corners of the largest contour in the image."""
	opencv_version = cv2.__version__.split('.')[0]
	if opencv_version == '3':
		_, contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
	else:
		contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
	contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by area, descending
	polygon = contours[0]  # Largest image

	# Use of `operator.itemgetter` with `max` and `min` allows us to get the index of the point
	# Each point is an array of 1 coordinate, hence the [0] getter, then [0] or [1] used to get x and y respectively.

	# Bottom-right point has the largest (x + y) value
	# Top-left has point smallest (x + y) value
	# Bottom-left point has smallest (x - y) value
	# Top-right point has largest (x - y) value
	bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

	# Return an array of all 4 points using the indices
	# Each point is in its own array of one coordinate
	return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]


def distance_between(p1, p2):
	"""Returns the scalar distance between two points"""
	a = p2[0] - p1[0]
	b = p2[1] - p1[1]
	return np.sqrt((a ** 2) + (b ** 2))


def crop_and_warp(img, crop_rect):
	"""Crops and warps a rectangular section from an image into a square of similar size."""

	# Rectangle described by top left, top right, bottom right and bottom left points
	top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]

	# Explicitly set the data type to float32 or `getPerspectiveTransform` will throw an error
	src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

	# Get the longest side in the rectangle
	side = max([
		distance_between(bottom_right, top_right),
		distance_between(top_left, bottom_left),
		distance_between(bottom_right, bottom_left),
		distance_between(top_left, top_right)
	])

	# Describe a square with side of the calculated length, this is the new perspective we want to warp to
	dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

	# Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
	m = cv2.getPerspectiveTransform(src, dst)

	# Performs the transformation on the original image
	return cv2.warpPerspective(img, m, (int(side), int(side)))


def infer_grid(img):
	"""Infers 81 cell grid from a square image."""
	squares = []
	side = img.shape[:1]
	side = side[0] / 9

	# Note that we swap j and i here so the rectangles are stored in the list reading left-right instead of top-down.
	for j in range(9):
		for i in range(9):
			p1 = (i * side, j * side)  # Top left corner of a bounding box
			p2 = ((i + 1) * side, (j + 1) * side)  # Bottom right corner of bounding box
			squares.append((p1, p2))
	return squares


def cut_from_rect(img, rect):
	"""Cuts a rectangle from an image using the top left and bottom right points."""
	return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]


def scale_and_centre(img, size, margin=0, background=0):
	"""Scales and centres an image onto a new background square."""
	h, w = img.shape[:2]

	def centre_pad(length):
		"""Handles centering for a given length that may be odd or even."""
		if length % 2 == 0:
			side1 = int((size - length) / 2)
			side2 = side1
		else:
			side1 = int((size - length) / 2)
			side2 = side1 + 1
		return side1, side2

	def scale(r, x):
		return int(r * x)

	if h > w:
		t_pad = int(margin / 2)
		b_pad = t_pad
		ratio = (size - margin) / h
		w, h = scale(ratio, w), scale(ratio, h)
		l_pad, r_pad = centre_pad(w)
	else:
		l_pad = int(margin / 2)
		r_pad = l_pad
		ratio = (size - margin) / w
		w, h = scale(ratio, w), scale(ratio, h)
		t_pad, b_pad = centre_pad(h)

	img = cv2.resize(img, (w, h))
	img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
	return cv2.resize(img, (size, size))


def find_largest_feature(inp_img, scan_tl=None, scan_br=None):
	"""
	Uses the fact the `floodFill` function returns a bounding box of the area it filled to find the biggest
	connected pixel structure in the image. Fills this structure in white, reducing the rest to black.
	"""
	img = inp_img.copy()  # Copy the image, leaving the original untouched
	height, width = img.shape[:2]

	max_area = 0
	seed_point = (None, None)

	if scan_tl is None:
		scan_tl = [0, 0]

	if scan_br is None:
		scan_br = [width, height]

	# Loop through the image
	for x in range(scan_tl[0], scan_br[0]):
		for y in range(scan_tl[1], scan_br[1]):
			# Only operate on light or white squares
			if img.item(y, x) == 255 and x < width and y < height:  # Note that .item() appears to take input as y, x
				area = cv2.floodFill(img, None, (x, y), 64)
				if area[0] > max_area:  # Gets the maximum bound area which should be the grid
					max_area = area[0]
					seed_point = (x, y)

	# Colour everything grey (compensates for features outside of our middle scanning range
	for x in range(width):
		for y in range(height):
			if img.item(y, x) == 255 and x < width and y < height:
				cv2.floodFill(img, None, (x, y), 64)

	mask = np.zeros((height + 2, width + 2), np.uint8)  # Mask that is 2 pixels bigger than the image

	# Highlight the main feature
	if all([p is not None for p in seed_point]):
		cv2.floodFill(img, mask, seed_point, 255)

	top, bottom, left, right = height, 0, width, 0

	for x in range(width):
		for y in range(height):
			if img.item(y, x) == 64:  # Hide anything that isn't the main feature
				cv2.floodFill(img, mask, (x, y), 0)

			# Find the bounding parameters
			if img.item(y, x) == 255:
				top = y if y < top else top
				bottom = y if y > bottom else bottom
				left = x if x < left else left
				right = x if x > right else right

	bbox = [[left, top], [right, bottom]]
	return img, np.array(bbox, dtype='float32'), seed_point


def extract_digit(img, rect, size):
    """Extracts a digit (if one exists) from a Sudoku square."""
    digit = cut_from_rect(img, rect)  # Get the digit box from the whole square

    # Use fill feature finding to get the largest feature in middle of the box
    h, w = digit.shape[:2]
    margin = int(np.mean([h, w]) / 2.5)
    _, bbox, seed = find_largest_feature(digit, [margin, margin], [w - margin, h - margin])
    digit = cut_from_rect(digit, bbox)

    # Check for empty cells or invalid bounding boxes
    non_zero_pixels = cv2.countNonZero(digit)
    if non_zero_pixels < 50:  # Threshold for empty cells
        return np.zeros((size, size), np.uint8)

    # Validate bounding box dimensions
    bbox_width = bbox[1][0] - bbox[0][0]
    bbox_height = bbox[1][1] - bbox[0][1]
    if bbox_height < bbox_width:  # If height is greater than width
        return np.zeros((size, size), np.uint8)  # Return a black square

    # Scale and pad the digit if it's valid
    if bbox_width > 0 and bbox_height > 0 and (bbox_width * bbox_height) > 100:
        return scale_and_centre(digit, size, 4)
    else:
        return np.zeros((size, size), np.uint8)

def get_digits(img, squares, size):
	"""Extracts digits from their cells and builds an array"""
	digits = []
	img = pre_process_image(img.copy(), skip_dilate=True)
	print("get digits")
	show_image(img)
	for square in squares:
		digits.append(extract_digit(img, square, size))
	return digits


def parse_grid(path):
	original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	processed = pre_process_image(original)
	
#    cv2.namedWindow('processed',cv2.WINDOW_AUTOSIZE)
#    processed_img = cv2.resize(processed, (500, 500))          # Resize image
	
	corners = find_corners_of_largest_polygon(processed)
	cropped = crop_and_warp(original, corners)
	
	cv2.namedWindow('cropped',cv2.WINDOW_AUTOSIZE)
	cropped_img = cv2.resize(cropped, (600, 600))              # Resize image
	cv2.imwrite('cropped_image.jpg', cropped_img)

	squares = infer_grid(cropped)
	
	
	print("Get digit")	
	digits = get_digits(cropped, squares, 28)
	print("Get Final img")	
	final_image, predicts = show_digits(digits)
	print("Predicts")
	print(predicts)
	return final_image

def output(a):
    sys.stdout.write(str(a))


def extract_sudoku(image_path):
    final_image = parse_grid(image_path)
    return final_image