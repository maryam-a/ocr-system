# ocr-system

MIT 6.819 Final Project - Optical Character Recognition

Authors: [Maryam Archie](marchie@mit.edu) and [Gloria Fang](yfang@mit.edu)

## Getting Started
1. Download Python 3 (if you don't already - preferably from Anaconda)

2. Install the following dependencies:
```
pip install numpy
pip install tensorflow
pip install tensorflow-gpu
pip install scipy
pip install pillow
pip install cv2
pip install sklearn
pip install skimage
pip install TFANN
```

3. Install the following fonts if you don't have them already:
```
consola.ttf
lucon.ttf
OCRAEXT.TTF
cour.ttf
```

4. Clone the repository.

5. Change any paths if necessary.

## Generating the data
You can choose to generate one of four fonts: Consola, Lucida Console, OCR-A Extended and Courier New.

In `generate_images.py`, change the font `TEXT_FONT` as desired. File paths and font sizes will updated appropriately.

You can also change the number of single-line and multi-line images by changing `N_IMAGES` or specifying the `num_images` argument in the `create_single_line_text_images` and `create_multiple_line_text_images` functions.

To generated the images, simply run the following command:
```
python generate_images.py
```

We have already generated 20,000 images for each font type. These are found in `data`.

## Training the model
To train the model, first you need to make sure that `TEXT_FONT` in `ocr.py` is the same as that in `generate_images.py`. Next, simply run the command below:
```
python ocr.py
```

When the model has been trained, it is saved so that the next time the script is run, it simply loads the model and makes its prediction.

## Making predictions
Run the following command to predict the text given an image:

#### Single Line
```
python ocr.py -s 'data/single_line/<text-font>/<image-name>.png'
```

#### Multiple Lines
```
python ocr.py -m 'data/multiple_lines/<text-font>/<image-name>.png'
```
