# Cosine Art

This is a script to create an image using frequency modulated sine waves, given an input image. It just uses numpy to process the image pixel by pixel. The image is first 

## Usage
The usage is as follows:
```bash
python3 cosine_art.py [options] -i input_image_path
```
All the command line options are as follows:
```
  -h, --help            show this help message and exit
  -i INPUT_FILE, --input-file INPUT_FILE
                        path to input image
  -o OUTPUT_FILE, --output-file OUTPUT_FILE
                        path to output image
  -c WAV_COLOR, --wav-color WAV_COLOR
                        color of the waveforms. It could either be a code in
                        the format e.g. #FF00FF or one of [black, white, red,
                        green, blue, yellow]
  -b BACKGROUND_COLOR, --background-color BACKGROUND_COLOR
                        background color, for fomart look at wav-color
  -w WIDTH, --width WIDTH
                        width of each waveform row in pixels
  -f {[0.1,0.4]}, --freq-factor {[0.1,0.4]}
                        frequency factor in range [0.1, 0.4] for wavforms,
                        higher the number, more compact the wavs
  --invert              invert colors, so that dark goes to less frequency and
                        vice-versa
```

The script as run on one of the test images provided in `test_images`:
```bash
python3 cosine_art.py -i test_images/test_image.jpg
```

## Formats supported
As of now, jpeg, png are supported, without any exhaustive testing. In case of any issues, or if you want more formats supported, create an issue.

## Test Image credits
Both test images are taken from [unsplash.com](unsplash.com). `test_image_1.jpg` is a photo by [Shelley Kim](https://unsplash.com/@shelleykim), and `test_image_2.jpg` is by [taylor hernandez](https://unsplash.com/@taylormae)
