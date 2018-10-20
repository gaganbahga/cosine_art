from scipy import misc
import matplotlib.pyplot as plt
import numpy as np

def main():
	img = misc.imread('test_image.jpg')
	img = img * np.asarray([0.299, 0.587, 0.114])
	img = np.sum(img, axis=2)
	img = np.max(img) - img

	img = subsample(img)
	img = get_cosine_img(img)
	img = -img + 1 

	misc.imsave('generated.png', img)

def get_cosine_img(img, row_width=10, col_width=1, max_wavs_perblock=0.1):
	img = img/img.max()
	new_img = np.zeros([row_width * img.shape[0], col_width * img.shape[1]])
	max_freq = 2 * np.pi * max_wavs_perblock/col_width
	img = max_freq * img

	for i in range(img.shape[0]):
		row = img[i]
		phase = 0
		for j in range(len(row)):
			f = row[j]
			if j == 0:
				f_prev = f
			else:
				f_prev = row[j - 1]
			block, phase = get_block(f, f_prev, max_freq, phase, row_width, col_width)
			new_img[i * row_width : (i+1)*row_width, j * col_width : (j+1)*col_width] = block
	return new_img


def get_block(f, f_prev, max_freq, phase, row_width, col_width, block_subdivision=1):
	freqs = np.linspace(f_prev, f, block_subdivision)
	block = np.zeros([row_width, col_width])
	block_widths = get_widths(col_width, block_subdivision)
	# print(block_widths)
	last_sample = row_width//2 + int((row_width//2) * np.sin(phase))
	for i, f in enumerate(freqs):
		y = 0.98*(row_width//2) * np.sin(f * np.arange(block_widths[i]) + phase)
		y = y + row_width//2
		y = y.astype(np.int32)
		# print(block_widths[i], row_width)
		sub_block = np.zeros((row_width, block_widths[i]))
		# sub_block[y.astype(np.int32), np.arange(block_widths[i])] = 1
		for j in range(block_widths[i]):
			if j != 0:
				prev = y[j-1]
			else:
				prev = last_sample
			
			if y[j] > prev:
				sub_block[prev:y[j], j] = 1
			else:
				sub_block[y[j]:prev, j] = 1
		last_sample = y[-1]
			# sub_block[y[j]:(y[j+1])//2, j] = 1
		# print(block_widths[:i],block_widths[:i+1])
		block[:, np.sum(block_widths[:i]):np.sum(block_widths[:i+1])] = sub_block
		phase = np.unwrap(np.asarray([phase + f * block_widths[i]]))[0]
	return block, phase

		
def get_widths(width, no_divisions):
	blocks = (width//no_divisions) * np.ones([no_divisions])
	left = width - (width//no_divisions) * no_divisions
	blocks[:left] += 1
	return blocks.astype(np.int32)


def subsample(img, row_factor=10, col_factor=1):
	num_rows = img.shape[0]//row_factor
	num_cols = img.shape[1]//col_factor
	margin_rows = img.shape[0] - num_rows * row_factor
	margin_cols = img.shape[1] - num_cols * col_factor
	img = img[margin_rows//2 : margin_rows//2 + num_rows * row_factor, 
		  	  margin_cols//2 : margin_cols//2 + num_cols * col_factor]

	new_img = np.zeros([num_rows, num_cols])
	for i in range(num_rows):
		for j in range(num_cols):
			new_img[i, j] = (np.median(img[i * row_factor : (i + 1) * row_factor,
								j * col_factor : (j + 1) * col_factor]))
	return new_img

if __name__ == '__main__':
	main()