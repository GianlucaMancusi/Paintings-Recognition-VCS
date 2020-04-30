import numpy as np
import cv2
import time
from colorama import init, Fore, Back, Style
init()
from image_viewer import ImageViewer
from stopwatch import Stopwatch

class Function:
	def __init__(self, function, multiwrapper=False, **kwargs):
		self.function = function
		self.params = kwargs
		self.multiwrapper = multiwrapper
	
	def run(self, debug=False, **kwargs):
		params = self.params.copy()
		add_params_to_dict(params, **kwargs)
		if self.multiwrapper:
			return self.multi_run(debug=debug, **params)
		else:
			return self.function(debug=debug, **params)
	
	def multi_run(self, debug=False, **kwargs):
		return multiwrapper(func=self.function, debug=debug, **kwargs)
	
	def addParams(self, **kwargs):
		add_params_to_dict(self.params, **kwargs)
	
	def __str__(self):
		return self.function.__name__

def add_params_to_dict(dictionay, **kwargs):
	for key, value in kwargs.items():
		dictionay[key] = value

def merge_sum_norm(images):
	canvas = np.zeros_like(images[0], dtype=np.int64)
	for img in images:
		canvas += img
	canvas = canvas / canvas.max()
	canvas *= 255
	return canvas.astype(np.uint8)

def merge_sum_clip(images):
	canvas = np.zeros_like(images[0], dtype=np.int64)
	for img in images:
		canvas += img
	np.clip(canvas, a_min=0, a_max=255,  out=canvas)
	return canvas.astype(np.uint8)

def merge_sum_threshold(images):
	canvas = merge_sum_norm(images)
	_, result = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY)
	return result

def multiwrapper(input: list, func, debug=False, multi_debug_merge_func=merge_sum_norm, **kwargs):
	output_list = []
	output_debug_list = []
	for single_input in input:
		if debug:
			output, output_debug = func(single_input, debug=debug, **kwargs)
			output_debug_list.append(output_debug)
		else:
			output = func(single_input, debug=debug, **kwargs)
		output_list.append(output)
	if debug:
		output_debug_merged = multi_debug_merge_func(output_debug_list)
		return output_list, output_debug_merged
	else:
		return output_list

class Pipeline:
	functions = []
	debug_out_list = []
	source = None

	def __init__(self, functions=[], default=False):
		self.functions = functions
		self.set_default() if default else None
		self.debug_out_list = []
	
	def set_default(self, load_first=None):
		self.functions = []
		if load_first is None or load_first > 0:
			from step_01_mean_shift_seg import mean_shift_segmentation
			self.functions.append(Function(mean_shift_segmentation, spatial_radius=7, color_radius=150, maximum_pyramid_level=1))

		if load_first is None or load_first > 1:
			from step_02_mask_largest_segment import mask_largest_segment
			self.functions.append(Function(mask_largest_segment, delta=48, x_samples=30))

		if load_first is None or load_first > 2:
			from step_03_dilate_invert import erode_dilate, invert, add_padding
			self.functions.append(Function(erode_dilate, size=5, erode=False))
			self.functions.append(Function(invert))
			self.functions.append(Function(add_padding, pad=100))

		if load_first is None or load_first > 3:
			from step_04_connected_components import find_contours
			self.functions.append(Function(find_contours))

		if load_first is None or load_first > 4:
			from step_05_find_paintings import find_possible_contours
			self.functions.append(Function(find_possible_contours))

		if load_first is None or load_first > 5:
			from step_06_erosion import  clean_frames_noise, mask_from_contour
			self.functions.append(Function(mask_from_contour, multiwrapper=True))
			self.functions.append(Function(clean_frames_noise, multiwrapper=True))

		if load_first is None or load_first > 6:
			from step_07_median_filter import apply_median_filter
			self.functions.append(Function(apply_median_filter, multiwrapper=True))

		if load_first is None or load_first > 7:
			from step_08_canny_edge_detection import apply_edge_detection
			self.functions.append(Function(apply_edge_detection, multiwrapper=True, multi_debug_merge_func=merge_sum_threshold))

		if load_first is None or load_first > 8:
			from step_09_hough import hough
			self.functions.append(Function(hough, multiwrapper=True, pad=100))

		if load_first is None or load_first > 9:
			from step_10_find_corners import find_corners
			self.functions.append(Function(find_corners, multiwrapper=True, multi_debug_merge_func=merge_sum_clip))
	
	def append(self, function):
		self.functions.append(function)
	
	def pop(self):
		self.functions.pop()

	def run(self, img, step=None, debug=False, print_time=False, filename='unknown'):
		self.debug_out_list = []
		self.source = img
		funcs_to_run = self.functions[:step]
		output = self.source
		stopwatch = Stopwatch()
		log_string = '{}\t{: <16}\t{:.04f}s   {:.02f} fps'
		for func in funcs_to_run:
			if debug:
				output, output_debug = func.run(debug, input=output)
				self.debug_out_list.append(output_debug)
			else:
				output = func.run(debug, input=output)
			t = stopwatch.round()
			if print_time:
				fps = 1 / t if t != 0 else 999
				print(log_string.format(filename, str(func), t, fps))
			
		t = stopwatch.total()
		fps = 1 / t if t != 0 else 999
		print(Fore.YELLOW + log_string.format(filename, 'TOTAL', t, fps) + Fore.RESET)
		print() if print_time else None

		if debug:
			return self.debug_out_list[-1]
		else:
			return output
	
	def debug_history(self):
		iv = ImageViewer()
		iv.remove_axis_values()
		iv.add(self.source, 'source', cmap='bgr')
		for img, func in zip(self.debug_out_list, self.functions):
			iv.add(img, str(func), cmap='bgr')
		return iv

if __name__ == "__main__":
	from step_11_highlight_painting import highlight_paintings
	# from step_12_b_create_outer_rect import mask as mask_b
	from data_test.standard_samples import TEST_PAINTINGS

	pipeline = Pipeline(default=True)

	iv = ImageViewer(cols=3)
	iv.remove_axis_values()

	plots = []
	for filename in TEST_PAINTINGS:
		img = np.array(cv2.imread(filename))
		pipeline.append(Function(highlight_paintings, source=img, pad=100))
		out = pipeline.run(img, debug=True, print_time=True, filename=filename)
		plots.append(pipeline.debug_history())
		iv.add(out, cmap='bgr')
		pipeline.pop()

	for plot in plots:
		plot.show()
	
	iv.show()