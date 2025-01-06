import numpy as np
import cv2
import operator
import numpy as np
from matplotlib import pyplot as plt
import sys
import tensorflow as tf

model = tf.keras.models.load_model('se_cnn_mnist_28x28.h5')

def show_image(img):	
	"""
	Description: Displays the image using OpenCV's imshow function and saves it to a file.
	Arguments:
		img (numpy.ndarray): The image to display and save.
	Returns: The same image (img) after displaying and saving.
	"""

	print(type(img))
	print(img.shape)
	cv2.imshow('image', img)  
	cv2.imwrite('images/gau_sudoku3.jpg', img)
	cv2.waitKey(0)  
	cv2.destroyAllWindows()  
	return img

def convert_when_colour(colour, img):
	"""
	Description: Converts the image to color if it is grayscale and the colour argument is a tuple representing a color.
	Arguments:
		colour (tuple): The color to use for conversion, must have 3 elements (e.g., (R, G, B)).
		img (numpy.ndarray): The input image that may be grayscale.
	Returns: The converted image in color if necessary.
	"""
	if len(colour) == 3:
		if len(img.shape) == 2:
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		elif img.shape[2] == 1:
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	return img

def pre_process_image(img, skip_dilate=False):
	"""
	Description: Performs Gaussian blurring, adaptive thresholding, and optionally dilation to enhance the main features of the image.
    Arguments:
        img (numpy.ndarray): The input image to process.
        skip_dilate (bool): If True, no dilation will be performed.
    Returns: The processed image.
	"""

	proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)

	proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

	proc = cv2.bitwise_not(proc, proc)

	if not skip_dilate:
		kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
		proc = cv2.dilate(proc, kernel)

	return proc

def find_corners_of_largest_polygon(img):
	"""
	Description: Finds the 4 extreme corners of the largest polygon in the image, typically used for identifying the Sudoku grid.
	Arguments:
		img (numpy.ndarray): The input binary image to extract contours from.
	Returns: A list of four points representing the corners of the largest contour.
	"""

	opencv_version = cv2.__version__.split('.')[0]
	if opencv_version == '3':
		_, contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
	else:
		contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
	contours = sorted(contours, key=cv2.contourArea, reverse=True)  
	polygon = contours[0]  

	bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))


	return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]


def distance_between(p1, p2):
	"""
	Description: Calculates the Euclidean distance between two points.
	Arguments:
		p1 (tuple): The first point (x, y).
		p2 (tuple): The second point (x, y).
	Returns: The scalar distance between the two points.
	"""
	a = p2[0] - p1[0]
	b = p2[1] - p1[1]
	return np.sqrt((a ** 2) + (b ** 2))


def crop_and_warp(img, crop_rect):
	"""
	Description: Crops a rectangular section from the image based on the provided corner points and warps it into a square.
	Arguments:
		img (numpy.ndarray): The input image.
		crop_rect (list): A list of four points representing the corners of the rectangle to be cropped.
	Returns: A warped square image corresponding to the cropped rectangle.
	"""

	top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]

	src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

	side = max([
		distance_between(bottom_right, top_right),
		distance_between(top_left, bottom_left),
		distance_between(bottom_right, bottom_left),
		distance_between(top_left, top_right)
	])

	dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

	m = cv2.getPerspectiveTransform(src, dst)

	return cv2.warpPerspective(img, m, (int(side), int(side)))

def parse_grid(path):
	"""
	Description: Parses a Sudoku grid from an image by preprocessing the image, finding the grid corners, and cropping and warping the image to extract the grid.
	Arguments:
		path (str): The path to the input Sudoku image.
	Returns: A cropped and warped image of the Sudoku grid.
	"""

	original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	processed = pre_process_image(original)
	
	corners = find_corners_of_largest_polygon(processed)
	cropped = crop_and_warp(original, corners)
	
	cropped_img = cv2.resize(cropped, (600, 600))

	cv2.imwrite('cropped_image.jpg', cropped_img)

	return cropped_img

def output(a):
    sys.stdout.write(str(a))


def extract_sudoku(image_path):
    final_image = parse_grid(image_path)
    return final_image