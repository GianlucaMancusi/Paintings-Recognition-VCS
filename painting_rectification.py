import cv2
import numpy as np
from image_viewer import ImageViewer
from painting_detection import painting_detection
from data_test.standard_samples import TEST_PAINTINGS

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

def perspective_dim(tl, tr, br, bl):	
	width_B = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	width_T = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	height_R = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	height_L = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

	if 0.0 in [width_B, width_T, height_R, height_L]:
		return None, None

	height_ratio = max(height_L / height_R, height_R / height_L)
	width_ratio = max(width_B / width_T, width_T / width_B)

	if 0.0 in [height_ratio, width_ratio]:
		return None, None

	width = max(width_B, width_T) * (height_ratio ** 2)
	height = max(height_L, height_R) * (width_ratio ** 2)

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
	x = x // len(pts)
	y = y // len(pts)
	return (x, y)

if __name__ == "__main__":
	for filename in TEST_PAINTINGS:
		rgbImage = cv2.imread(filename)
		painting_contours = painting_detection(rgbImage)
		rgbImage_num = rgbImage.copy()
		for i, corners in enumerate(painting_contours):
			rgbImage_num = cv2.putText(rgbImage_num, str(i + 1), mean_center(corners), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255) , 20, cv2.LINE_AA) 
		iv = ImageViewer()
		iv.add(rgbImage_num, 'original', cmap='bgr')
		for i, corners in enumerate(painting_contours):
			img = four_point_transform(rgbImage, np.array(corners))
			if not img is None:
				iv.add(img, 'painting {}'.format(i + 1), cmap='bgr')
		iv.show()

