#!/bin/bash/python3

# cosine_art.py
# author: Gagandeep Singh, 19 Oct, 2018

from os import path
import imageio.v2 as imageio
import argparse
from tqdm import tqdm
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom

# --- SVG generation support ---
def generate_svg(img, row_width, freq_factor, wav_color, bg_color, output_file):
    """
    Generate an SVG file with cosine curves modulated by the image.
    :param img: np.ndarray, 2-D image array
    :param row_width: int, width of each wave row in pixels
    :param freq_factor: float, frequency scaling factor for the cosine modulation
    :param wav_color: (int, int, int), RGB tuple for the waveform color
    :param bg_color: (int, int, int), RGB tuple for the background color
    :param output_file: str, path to the output SVG file
    """
    img = img / img.max()
    max_freq = 2 * np.pi * freq_factor
    img = max_freq * img
    height = row_width * img.shape[0]
    width = img.shape[1]

    svg = ET.Element('svg', xmlns="http://www.w3.org/2000/svg", width=str(width), height=str(height), version="1.1")
    # Background
    bg_hex = '#%02x%02x%02x' % bg_color
    ET.SubElement(svg, 'rect', x="0", y="0", width=str(width), height=str(height), fill=bg_hex)

    wav_hex = '#%02x%02x%02x' % wav_color
    for i in tqdm(range(img.shape[0]), desc='Generating SVG rows'):
        row = img[i]
        phase = 0
        points = []
        for j in range(len(row)):
            f = row[j]
            if j == 0:
                f_prev = f
            else:
                f_prev = row[j - 1]
            block, phase = get_column(f, f_prev, phase, row_width)
            # Find the y position of the wave (where block==1)
            y_indices = np.where(block == 1)[0]
            if len(y_indices) > 0:
                y = y_indices[0] + i * row_width
            else:
                y = (row_width // 2) + i * row_width
            points.append(f"{j},{y}")
        # Draw polyline for this row
        ET.SubElement(svg, 'polyline', points=' '.join(points), stroke=wav_hex, fill="none", stroke_width="1")

    # Pretty print and write
    rough_string = ET.tostring(svg, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_svg = reparsed.toprettyxml(indent="  ")
    with open(output_file, 'w') as f:
        f.write(pretty_svg)

def main(args):
    img = np.asarray(imageio.imread(args.input))
    if len(img.shape) == 3 and img.shape[-1] == 4: # last dimension is alpha
        img = img[:, :, :3]
    if len(img.shape) == 3 and img.shape[-1] == 3:
        img = img * np.asarray([0.299, 0.587, 0.114]) # convert to greyscale
        img = np.sum(img, axis=2)
    if not args.invert:
        img = np.max(img) - img

    row_width = img.shape[0]//args.num_rows
    freq_factor = args.freq_factor * 0.3 + 0.1 # just scaling

    img = subsample(img, args.num_rows)
    wav_color = get_color(args.wav_color)
    bg_color = get_color(args.background_color)

    output_file = args.output_file
    if output_file is None:
        output_file = path.join(path.dirname(args.input), 'generated.svg' if getattr(args, 'svg', False) else 'generated.png')

    if args.svg:
        generate_svg(img, row_width, freq_factor, wav_color, bg_color, output_file)
        print('Generated SVG image is present at', output_file)
    else:
        img = get_cosine_img(img, row_width, freq_factor)
        final_img = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')

        for i in range(3):
            final_img[:,:,i][img == 0] = bg_color[i]
            final_img[:,:,i][img == 1] = wav_color[i]

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


def subsample(img, num_rows):
    """
    Subsample the image, averaging over num_rows
    :param img: np.ndarray, 2-D image array
    :param num_rows: int, subsample per num_rows
    """
    row_width = img.shape[0]//num_rows

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
    parser.add_argument('input',
                        type=str, help='path to input image')
    parser.add_argument('-o', '--output-file', default=None,
                        type=str, help='path to output image')
    parser.add_argument('-c', '--wav-color', default='black',
                        type=str, help='color of the waveforms. It could either be a code in the format \
                        e.g. #FF00FF or one of [black, white, red, green, blue, yellow]')
    parser.add_argument('-b', '--background-color', default='white',
                        type=str, help='background color, for format look at wav-color')
    parser.add_argument('-n', '--num-rows', default=100,
                        type=int, help='number of waveforms (rows) in the generated image')
    parser.add_argument('-f', '--freq-factor', default=1/3,
                        type=float, choices=[Range(0., 1.)],
                        help='frequency factor in range [0., 1.] for wavforms, higher the number, more compact the wavs')
    parser.add_argument('--invert', action='store_true',
                         help='invert colors, so that dark goes to less frequency and vice-versa')
    parser.add_argument('--svg', action='store_true', default=False,
                        help='output SVG instead of PNG')
    args = parser.parse_args()
    main(args)
