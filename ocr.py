# Author: Maryam Archie and Gloria (Yung Wei Fang)

import os
import sys
import getopt
import string
import numpy as np
from skimage.io import imread
from sklearn import model_selection
from TFANN import ANNC
import cuts_lines

# Setting the seed
np.random.seed(123)

# Constant Values
DATA_ROOT = 'data/'
SL_DIR = DATA_ROOT + 'single_line/'
ML_DIR = DATA_ROOT + 'multiple_lines/'
DEMO_DIR = DATA_ROOT + 'demo/'
SL_FILE = DATA_ROOT + 'sl_data.txt'
ML_FILE = DATA_ROOT + 'ml_data.txt'
IMAGE_ENCODING = '.png'
MAX_CHARS = 64
NUM_CHARS = len(string.ascii_letters + string.digits + ' ')
IMAGE_SIZE = (14, 640, 3) # image size for the CNN

TEXT_INFO = {
    'consola.ttf': {'size': 18, 'dir': 'consola/', 'model': 'model-consolas'},
    'cour.ttf': {'size': 16, 'dir': 'cour/', 'model': 'model-cour'},
    'lucon.ttf': {'size': 17, 'dir': 'lucon/', 'model': 'model-lucon'},
    'OCRAEXT.TTF': {'size': 16, 'dir': 'ocr-a/', 'model': 'model-ocr-a'}
}

TEXT_FONT = 'lucon.ttf' # Change this
TEXT_SIZE = TEXT_INFO[TEXT_FONT]['size']
CURRENT_SL_DIR = SL_DIR + TEXT_INFO[TEXT_FONT]['dir']
CURRENT_ML_DIR = ML_DIR + TEXT_INFO[TEXT_FONT]['dir']
CURRENT_DEMO_DIR = DEMO_DIR + TEXT_INFO[TEXT_FONT]['dir']
CURRENT_MODEL_DIR = TEXT_INFO[TEXT_FONT]['model']
NN_DIR = 'ocr-nn'
MODEL_CLASSES = CURRENT_MODEL_DIR + '/_classes.txt'

ITERS = 500

def load_data(train_dir=CURRENT_SL_DIR, train_file=SL_FILE):
    '''
    Loads and prepares the dataset for training and testing.
    train_dir: string, The path to directory with the training images
    train_file: string, The file path to the image and ground truth

    Returns:
    Image Matrix, Ground Truth with Padding Matrix, Original Ground Truth Matrix,
    Image File Name Matrix
    '''
    images, gt_padding, gt, image_names = [], [], [], []

    with open(train_file) as train_data:
        for line in train_data:
            filename, text = line.strip().split(IMAGE_ENCODING + ' ')
            filename = filename + IMAGE_ENCODING
            padded_text = text.ljust(MAX_CHARS, ' ')

            images.append(imread(train_dir + filename))
            gt_padding.append(list(padded_text))
            gt.append(text)
            image_names.append(filename)

    return np.stack(images), np.stack(gt_padding), np.stack(gt), np.stack(image_names)

def accuracy(predictions, ground_truth):
    '''
    Calculates the accuracy of the predictions compared to the ground truths
    predictions: np.array, The text predictions for a set of images
    ground_truth: np.array, The ground truth text for a set of images
    '''
    result = sum(sum(i == j for i, j in zip(guess, truth)) / len(guess) 
                            for guess, truth in zip(predictions, ground_truth)) / len(predictions)
    return result

def predict_string_from_image(image):
    '''
    Given an image, the OCR system will predict the text embedded in the image
    image: ndarray, The image from which we want the text
    '''
    split_images = split_image(image)
    prediction = cnnc.predict(split_images)

    _, height_prop, width_prop = get_split_image_shape(image)
    size = (height_prop, width_prop)

    return arrange_text(prediction, size)

def split_image(image):
    '''
    Splits an image into smaller pieces for the neural network
    image: ndarray, The image from which we want the text
    '''
    channels, height_prop, width_prop = get_split_image_shape(image)
    new_height = height_prop * IMAGE_SIZE[0]
    new_width = width_prop * IMAGE_SIZE[1]

    # Get rid of excess pixels
    image = image[0:new_height, 0:new_width]

    # Reshape image
    image = image.reshape(height_prop, IMAGE_SIZE[0], -1, IMAGE_SIZE[1], channels)
    image = image.swapaxes(1,2).reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], channels)
    return image

def get_split_image_shape(image):
    '''
    Determines the height and width of the split image
    image: ndarray, The image from which we want the text
    '''
    height, width, channels = image.shape
    height_prop = height // IMAGE_SIZE[0]
    width_prop = width // IMAGE_SIZE[1]
    return channels, height_prop, width_prop

def arrange_text(text, size):
    '''
    Rearranges the predicted text based on the position of the split images
    text: string, The predicted text
    size: (height, width), The size of each of the split images
    '''
    predicted_text = np.array([''.join(p_text) for p_text in text])
    predicted_text = predicted_text.reshape(size)
    return '\n'.join(''.join(pp_text for pp_text in p_text) for p_text in predicted_text)

def load_multiline_data(filepath=ML_FILE):
    '''
    Load ground truths about images into a dictionary
    filepath: string, The path to the directory with the images of interest
    '''
    ml_data = {}
    with open(filepath, newline='') as ml_file:
        for line in ml_file:
            filename, text = line.strip().split(IMAGE_ENCODING + ' ')
            filename = filename + IMAGE_ENCODING
            ml_data[filename] = text
    return ml_data

def predict_multiline(a):
    '''
    '''
    ml_data = load_multiline_data()
    print('\nThis is the original text')
    image_name = a.split(CURRENT_ML_DIR)[1]
    text = ml_data[image_name]
    num_lines = len(text.split('\\n'))
    print(text.replace("\\n", "\n"))
    cuts_lines.slice_image(a, 'm')

    for i in range(num_lines):
        img_path = 'test/image' + image_name[:-4] + '-' + str(i) + '.png'
        image = imread(img_path)
        prediction = predict_string_from_image(image)
        print(prediction)

def predict_demo(a):
    image_name = a.split(DEMO_DIR)[1]
    with open('data/demo.txt') as demo_text:
        for line in demo_text:
            print(line.strip())
    cuts_lines.slice_image(a, 'd')

    print("\nThis is our guess")
    for i in range(18):
        image = imread('test/demo-' + str(i) + '.png')
        prediction = predict_string_from_image(image)
        print(prediction)

# Architecture for the Convolutional Neural Network
architecture = [('C', [4, 4,  3, NUM_CHARS // 2], [1, 2, 2, 1]), ('AF', 'relu'), 
                ('C', [4, 4, NUM_CHARS // 2, NUM_CHARS], [1, 2, 1, 1]), ('AF', 'relu'), 
                ('C', [8, 5, NUM_CHARS, NUM_CHARS], [1, 8, 5, 1]), ('AF', 'relu'),
                ('R', [-1, 64, NUM_CHARS])]

# Building the model
cnnc = ANNC(IMAGE_SIZE, architecture, batchSize = 64, learnRate = 5e-5, maxIter = ITERS, reg = 1e-5, tol = 1e-2, verbose = True)
if not cnnc.RestoreModel(CURRENT_MODEL_DIR + '/', NN_DIR):
    images, gt_padding, gt, image_names = load_data()

    # Shuffle and split data
    shuffled_data = model_selection.ShuffleSplit(n_splits = 1, random_state = 123)
    train, test = next(shuffled_data.split(images))

    # Fit data to neural network
    cnnc.fit(images[train], gt_padding[train])

    # Retrieve the predictions as a sequence of character indices
    prediction_seq = []
    for i in np.array_split(np.arange(images.shape[0]), 32): # 32 x 32
        prediction_seq.append(cnnc.predict(images[i]))
    prediction_seq = np.vstack(prediction_seq)

    # Convert the predictions from a sequence to strings
    prediction_string = np.array([''.join(seq) for seq in prediction_seq])

    # Compute accuracy
    train_acc = accuracy(prediction_string[train], gt[train])
    test_acc = accuracy(prediction_string[test], gt[test])
    print('\nTrain accuracy: ' + str(train_acc))
    print('Test accuracy: ' + str(test_acc))

    # Debug: Show the ground truth and the predicted text
    # for guess, text, name in zip(prediction_string, gt, image_names):
    #     print(name + ': ' + text + ' -> ' + guess)

    # Save model for next time
    cnnc.SaveModel(os.path.join(CURRENT_MODEL_DIR, NN_DIR))
    with open(MODEL_CLASSES, 'w') as classes_file:
        classes_file.write('\n'.join(cnnc._classes))

else:
    with open(MODEL_CLASSES) as classes_file:
        cnnc.RestoreClasses(classes_file.read().splitlines())


if __name__ == "__main__":
    options, args = getopt.getopt(sys.argv[1:], 's:m:d:')

    for o, a in options:
        if o == '-s':
            image = imread(a)
            prediction = predict_string_from_image(image)

        elif o == '-m':
            predict_multiline(a)

        elif o == '-d':
            predict_demo(a)

        else:
            print("Usage: %s -s singleline -m multiline -d demo" % sys.argv[0])

# Input files should be 'data/multiple_lines/x.png'     

# # python3 ocr.py 'data/demo/demo.png'

# # TODO: actually go through order
# # TODO: make the other images