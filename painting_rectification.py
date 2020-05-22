import cv2
import numpy as np
from image_viewer import ImageViewer
from painting_detection import painting_detection
import math

def square(x):
	return x * x

def aspect_ratio(corners, img):
	tl, tr, br, bl = order_points(corners)
	m1x, m1y = bl
	m2x, m2y = br
	m3x, m3y = tl
	m4x, m4y = tr
	v0 = img.shape[0] / 2
	u0 = img.shape[1] / 2
	# in case it matters: licensed under GPLv2 or later
	# legend:
	# square(x)  = x*x
	# sqrt(x) = square root of x

	# let m1x,m1y ... m4x,m4y be the (x,y) pixel coordinates
	# of the 4 corners of the detected quadrangle
	# i.e. (m1x, m1y) are the cordinates of the first corner, 
	# (m2x, m2y) of the second corner and so on.
	# let u0, v0 be the pixel coordinates of the principal point of the image
	# for a normal camera this will be the center of the image, 
	# i.e. u0=IMAGEWIDTH/2; v0 =IMAGEHEIGHT/2
	# This assumption does not hold if the image has been cropped asymmetrically

	# first, transform the image so the principal point is at (0,0)
	# this makes the following equations much easier
	m1x -= u0
	m1y -= v0
	m2x -= u0
	m2y -= v0
	m3x -= u0
	m3y -= v0
	m4x -= u0
	m4y -= v0


	# temporary variables k2, k3
	k2 = ((m1y - m4y)*m3x - (m1x - m4x)*m3y + m1x*m4y - m1y*m4x) / ((m2y - m4y)*m3x - (m2x - m4x)*m3y + m2x*m4y - m2y*m4x)
	k3 = ((m1y - m4y)*m2x - (m1x - m4x)*m2y + m1x*m4y - m1y*m4x) / ((m3y - m4y)*m2x - (m3x - m4x)*m2y + m3x*m4y - m3y*m4x)

	# f_squared is the focal length of the camera, squared
	# if k2==1 OR k3==1 then this equation is not solvable
	# if the focal length is known, then this equation is not needed
	# in that case assign f_squared= square(focal_length)
	f_squared = -((k3*m3y - m1y)*(k2*m2y - m1y) + (k3*m3x - m1x)*(k2*m2x - m1x)) / ((k3 - 1)*(k2 - 1))

	#The width/height ratio of the original rectangle
	whRatio = np.sqrt((square(k2 - 1) + square(k2*m2y - m1y)/f_squared + square(k2*m2x - m1x)/f_squared) / (square(k3 - 1) + square(k3*m3y - m1y)/f_squared + square(k3*m3x - m1x)/f_squared))

	# if k2==1 AND k3==1, then the focal length equation is not solvable 
	# but the focal length is not needed to calculate the ratio.
	# I am still trying to figure out under which circumstances k2 and k3 become 1
	# but it seems to be when the rectangle is not distorted by perspective, 
	# i.e. viewed straight on. Then the equation is obvious:
	if k2 == 1 or k3 == 1:
		whRatio = np.sqrt((square(m2y-m1y) + square(m2x-m1x)) / (square(m3y-m1y) + square(m3x-m1x)))

	# After testing, I found that the above equations 
	# actually give the height/width ratio of the rectangle, 
	# not the width/height ratio. 
	# If someone can find the error that caused this, 
	# I would be most grateful.
	# until then:
	return whRatio


def painting_rectification(img, corners):
	rect = order_points(corners)
	tl, tr, br, bl = rect
	whRatio = aspect_ratio(corners, img)
	if math.isnan(whRatio):
		return None
	width, height = perspective_dim_ratio(whRatio, tl, tr, br, bl)
	dst = np.array([
		[0, 0],
		[height - 1, 0],
		[height - 1, width - 1],
		[0, width - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(img, M, (height, width))
	# return the warped image
	return warped


def order_points(pts):
	pts = pts.squeeze()
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def perspective_dim_ratio(whRatio, tl, tr, br, bl):
	height_R = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	height_L = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	height = max(height_L, height_R)
	width = height * whRatio
	return int(height), int(width)

def perspective_dim(tl, tr, br, bl):	
	width_B = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	width_T = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	height_R = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	height_L = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

	if 0.0 in [width_B, width_T, height_R, height_L]:
		return None, None

	# height_ratio = max(height_L / height_R, height_R / height_L)
	# width_ratio = max(width_B / width_T, width_T / width_B)

	# if 0.0 in [height_ratio, width_ratio]:
	# 	return None, None

	# width = max(width_B, width_T) * (height_ratio ** 1)
	# height = max(height_L, height_R) * (width_ratio ** 1)

	# return int(height), int(width)
	height_ratio = max(height_L / height_R, height_R / height_L) - 1
	width_ratio = max(width_B / width_T, width_T / width_B) - 1

	if 0.0 in [height_ratio, width_ratio]:
		return None, None
	max_width = max(width_B, width_T)
	max_height = max(height_L, height_R)
	width = np.sqrt(max_width ** 2 * (1 + height_ratio ** 2))
	height = np.sqrt(max_height ** 2 * (1 + width_ratio ** 2))

	return int(height), int(width)

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	# rect = pts
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	width, height = perspective_dim(tl, tr, br, bl)
	if width is None or height is None:
		return None
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[height - 1, 0],
		[height - 1, width - 1],
		[0, width - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (height, width))
	# return the warped image
	return warped

def remove_pad(pts, pad):
	return [ [x - pad, y - pad] for x, y in pts ]

def mean_center(pts):
	x, y = 0, 0
	for pt in pts:
		x += pt[0, 0]
		y += pt[0, 1]
	x = x # len(pts)
	y = y # len(pts)
	return (x, y)

if __name__ == "__main__":
	from data_test.standard_samples import TEST_PAINTINGS, PERSPECTIVE
	from image_viewer import show_me
	from step_10_find_corners import draw_corners
	rgbImage = cv2.imread(PERSPECTIVE)
	for filename in TEST_PAINTINGS:
		# rgbImage = cv2.imread(filename)
		rgbImage = cv2.rotate(rgbImage, cv2.ROTATE_90_CLOCKWISE)
		painting_contours = painting_detection(rgbImage, area_perc=0.96)
		rgbImage_num = rgbImage.copy()
		for i, corners in enumerate(painting_contours):
			rgbImage_num = cv2.putText(rgbImage_num, str(i + 1), mean_center(corners), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255) , 20, cv2.LINE_AA) 
		for i, corners in enumerate(painting_contours):
			iv = ImageViewer()
			iv.add(rgbImage_num, 'original', cmap='bgr')
			img = four_point_transform(rgbImage, np.array(corners))
			if not img is None:
				iv.add(img, 'VIT painting {}'.format(i + 1), cmap='bgr')
			img = painting_rectification(rgbImage, np.array(corners))
			if not img is None:
				iv.add(img, 'STACK painting {}'.format(i + 1), cmap='bgr')
			iv.show()

