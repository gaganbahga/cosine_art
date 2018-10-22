#!/bin/bash/python3

# cosine_art.py
# author: Gagandeep Singh, 19 Oct, 2018

from os import path
import imageio
import argparse
from tqdm import tqdm
import numpy as np

def main(args):
	img = np.asarray(imageio.imread(args.input_file))
	if len(img.shape) == 3 and img.shape[-1] == 4: # last dimension is alpha
		img = img[:, :, :3]
	if len(img.shape) == 3 and img.shape[-1] == 3:
		img = img * np.asarray([0.299, 0.587, 0.114]) # convert to greyscale
		img = np.sum(img, axis=2)
	if not args.invert:
		img = np.max(img) - img

	img = subsample(img, args.width)
	img = get_cosine_img(img, args.width, args.freq_factor)
	wav_color = get_color(args.wav_color)
	bg_color = get_color(args.background_color)

	final_img = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')

	for i in range(3):
		final_img[:,:,i][img == 0] = bg_color[i]
		final_img[:,:,i][img == 1] = wav_color[i]

	output_file = args.output_file
	if output_file is None:
		output_file = path.join(path.dirname(args.input_file), 'generated.png')
	imageio.imwrite(output_file, final_img.astype('uint8'))
	print('Generated image is present at', output_file)

def get_color(color):
	"""
	Given a color code or color name, return a tuple of RGB values
	:param color: str, color code or name
	:return: (int, int, int), RGB tuple
	"""
	def decode_rgb_code(code):
		"""
		Given a color code, return a tuple of RGB values
		"""
		r = int(code[:2], 16)
		g = int(code[2:4], 16)
		b = int(code[4:], 16)
		return (r, g, b)

	if color[0] == '#':
		assert len(color) == 7, 'invalid color code'
		return decode_rgb_code(color[1:])
	else:
		if color.lower() == 'black':
			return (0, 0, 0)
		elif color.lower() == 'white':
			return (255, 255, 255)
		elif color.lower() == 'red':
			 return (255, 0, 0)
		elif color.lower() == 'green':
			return (0, 255, 255)
		elif color.lower() == 'blue':
			return (0, 0, 255)
		elif color.lower() == 'yellow':
			return (255, 255, 0)
		else:
			raise ValueError('color name not recognized')


def get_cosine_img(img, row_width, freq_factor):
	"""
	Main function for getting cosine image
	:param img: np.ndarray, shape=[length, width] image array
	:param row_width: int, width of each wave row in pixels
	:param freq_factor: float, factor for getting frequency corresponding to internsity
	:return: np.ndarray, same shape as img, converted cosine image 
	"""
	img = img/img.max()
	new_img = np.zeros([row_width * img.shape[0], img.shape[1]])
	max_freq = 2 * np.pi * freq_factor
	img = max_freq * img

	for i in tqdm(range(img.shape[0])):
		row = img[i]
		phase = 0
		for j in range(len(row)):
			f = row[j]
			if j == 0:
				f_prev = f
			else:
				f_prev = row[j - 1]
			block, phase = get_column(f, f_prev, phase, row_width)
			new_img[i * row_width : (i+1)*row_width, j : j+1] = np.expand_dims(block, axis=1)
	return new_img


def get_column(f, f_prev, phase, row_width):
	"""
	Get the values for a particular column of a row of the image
	:param f: float, "frequency" of this column
	:param f_prev: float, freequncy of immediate previous column
	:param phase: float, phase at the current column for starting point
	:param row_width: int, width of each wave row in pixels
	:return: np.ndarray, 1-D array equal to row-width (or column height)
	"""
	block = np.zeros(row_width)
	last_sample = row_width//2 + int((row_width//2) * np.sin(phase)) - 1
	y = (row_width//2) * np.sin(f + phase)
	y = y + row_width//2
	y = y.astype(np.int32)
	block[last_sample] = 1

	if y > last_sample:
		block[last_sample:y] = 1
	else:
		block[y:last_sample] = 1

	phase = np.unwrap(np.asarray([phase + f ]))[0]
	return block, phase


def subsample(img, row_width):
	"""
	Subsample the image, reducing number of rows
	:param img: np.ndarray, 2-D image array
	:param row_width: int, subsample the rows by that amount
	"""
	num_rows = img.shape[0]//row_width
	num_cols = img.shape[1]
	margin_rows = img.shape[0] - num_rows * row_width
	img = img[margin_rows//2 : margin_rows//2 + num_rows * row_width, :]

	new_img = np.zeros([num_rows, num_cols])
	for i in range(num_rows):
		new_img[i, :] = (np.median(img[i * row_width : (i + 1) * row_width, :], axis=0))
	return new_img


if __name__ == '__main__':
	class Range(object):
	    def __init__(self, start, end):
	        self.start = start
	        self.end = end
	    def __eq__(self, other):
	        return self.start <= other <= self.end
	    def __str__(self):
	    	return '[' + str(self.start) + ',' + str(self.end) + ']'
	        
	parser = argparse.ArgumentParser('Create image using cosine curves')
	parser.add_argument('-i', '--input-file', required=True,
						type=str, help='path to input image')
	parser.add_argument('-o', '--output-file', default=None,
						type=str, help='path to output image')
	parser.add_argument('-c', '--wav-color', default='black',
						type=str, help='color of the waveforms. It could either be a code in the format \
						e.g. #FF00FF or one of [black, white, red, green, blue, yellow]')
	parser.add_argument('-b', '--background-color', default='white',
						type=str, help='background color, for fomart look at wav-color')
	parser.add_argument('-w', '--width', default=15,
						type=int, help='width of each waveform row in pixels')
	parser.add_argument('-f', '--freq-factor', default=0.2,
						type=float, choices=[Range(0.1, 0.4)],
						help='frequency factor in range [0.1, 0.4] for wavforms, higher the number, more compact the wavs')
	parser.add_argument('--invert', action='store_true',
					 	help='invert colors, so that dark goes to less frequency and vice-versa')
	args = parser.parse_args()
	main(args)
