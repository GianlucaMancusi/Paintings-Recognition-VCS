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
			output_debug_list.append(output_debug) if not output_debug is None else None
		else:
			output = func(single_input, debug=debug, **kwargs)
		output_list.append(output)
	if debug:
		if len(output_debug_list) > 0:
			output_debug_merged = multi_debug_merge_func(output_debug_list)
		else:
			output_debug_merged = np.zeros((100, 100), np.uint8)
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
		"""
		load_first defines how many steps to load, if left None loads them all
		"""
		self.functions = []
		if load_first is None or load_first > 0:
			from step_01_pre_processing import mean_shift_segmentation
			self.functions.append(Function(mean_shift_segmentation))

		if load_first is None or load_first > 1:
			from step_02_background_detection import mask_largest_segment
			self.functions.append(Function(mask_largest_segment))

		if load_first is None or load_first > 2:
			from step_03_cleaning import opening, invert, add_padding
			self.functions.append(Function(opening))
			self.functions.append(Function(invert))
			self.functions.append(Function(add_padding, pad=100))

		if load_first is None or load_first > 3:
			from step_04_components_selection import find_contours
			self.functions.append(Function(find_contours))

		if load_first is None or load_first > 4:
			from step_04_components_selection import find_possible_contours
			self.functions.append(Function(find_possible_contours))

		if load_first is None or load_first > 5:
			from step_05_contour_pre_processing import  clean_frames_noise, mask_from_contour
			self.functions.append(Function(mask_from_contour, multiwrapper=True))
			self.functions.append(Function(clean_frames_noise, multiwrapper=True))

		if load_first is None or load_first > 6:
			from step_05_contour_pre_processing import apply_median_filter
			self.functions.append(Function(apply_median_filter, multiwrapper=True))

		if load_first is None or load_first > 7:
			from step_05_contour_pre_processing import apply_edge_detection
			self.functions.append(Function(apply_edge_detection, multiwrapper=True, multi_debug_merge_func=merge_sum_threshold))

		if load_first is None or load_first > 8:
			from step_06_corners_detection import hough
			self.functions.append(Function(hough, multiwrapper=True, pad=100))

		if load_first is None or load_first > 9:
			from step_06_corners_detection import find_corners
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
	from step_07_highlight_painting import highlight_paintings
	# from step_12_b_create_outer_rect import mask as mask_b
	from data_test.standard_samples import TEST_PAINTINGS, FISH_EYE, PEOPLE

	# Creando instanziando la classe Pipeline è possibile passare come valori
	# la lista delle funzioni che verranno eseguite, passando come valore default=True
	# verranno impostate come pipeline di funzioni quelle di default 
	pipeline = Pipeline(default=True)

	# l'ImageViewer adesso non ha più bisogno a priori del numero di immagini che verranno inserite
	iv = ImageViewer(cols=3)
	iv.remove_axis_values()

	plots = []
	# ho creato uno script per avere i filename delle immagini che usiamo come test
	# for filename in TEST_PAINTINGS:
	filename = "data_test/painting_09/00_calibration.jpg"
	img = np.array(cv2.imread(filename))
	# Tramite il comando append è possibile aggiungere una funzione alla pipeline,
	# in questo caso devo farlo perchè l'ultima funzione prende come source img
	pipeline.append(Function(highlight_paintings, source=img, pad=100))
	# tramite il comando run eseguo le funzioni in ordine.
	# con debug=True per ogni step vengono create delle immagini di debug che poi
	# possono essere visualizzate in sequenza
	# con print_time=True vengono stampati i tempi per ogni funzione
	# filename è opzionale, serve per la stampa
	out = pipeline.run(img, debug=True, print_time=True, filename=filename)
	# debug_history() ritorna una classe ImageViewer con la sequenza degli output di ogni funzione
	plots.append(pipeline.debug_history())
	iv.add(out, cmap='bgr')
	# con pop viene tolta l'ultima funzione dalla lista della pipeline
	pipeline.pop()

	for plot in plots:
		# vengono visualizzati tutti i grafici di debug
		plot.show()
	
	iv.show()

	# IMPORTANTE #######################################################################################
	# La Pipeline eseguita con il comando debug=False ritornerà i valori dell'ultima funzione che non sono
	# obbligatoriamente delle immagini, infatti in paintings_rectification.py la uso per ottenere i corners
	# dei quadri trovati. Mentre con debug=True ritornerà l'ultima immagine di debug creata.
	# 
	# Ogni funzione che viene inserita nella pipeline accetta una variabile input ed una variabile debug
	# In caso di debug=False ritorna un output, in caso di debug=True ritorna un output ed un immagine che
	# serve a far visualizzare il risultato della funzione.
	# 
	# La funzione viene "wrappata" dalla classe Function in modo da essere passata alla pipeline
	# La classe Function ha una variabile chiamata multiwrapper che serve per simulare i cicli for
	# Infatti prendendo come esempio la funzione "find_possible_contours" questa ritornerà una lista di
	# (img, contours). Con multiwrapper=True la funzione successiva "mask_from_contour" verra eseguita
	# per ogni elemento della lista e l'output totale sarà una concatenazione degli output. In caso di
	# debug=True l'immagine generata sarà una sorta di blending delle immagini di output generate dalla
	# singola funzione.