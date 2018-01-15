# 10.01.2017 Test task for CVT:
import methods
import glob
import os
import cv2
import string
import fnmatch
import numpy as np

original_image = None
mask_image = None
comparing_image = None
filenames = [img for img in glob.glob("Dataset/Dirty/*.jpg")]
filenames.sort()
name = []

number_of_symbols_total = 0
number_of_symbols_true_total = 0
number_of_images = 0
number_of_images_true = 0


for file in filenames:
    original_image = None
    mask_image = None
    comparing_image = None
    base = os.path.basename(file)
    name = os.path.splitext(base)[0]
    if name.isdigit():
        number_of_images += 1
        original_image = cv2.imread(file, 0)
        for file1 in filenames:
            base1 = os.path.basename(file1)
            name1 = os.path.splitext(base1)[0]
            if fnmatch.fnmatch(name1, name + "C"):
                comparing_image = cv2.imread(file1, 0)
            elif fnmatch.fnmatch(name1, name + "M"):
                mask_image = cv2.imread(file1, 0)
        if comparing_image is not None and mask_image is not None:
            number_of_symbols, number_of_symbols_true = methods.segment(original_image, mask_image, comparing_image, name)
            number_of_symbols_total = number_of_symbols_total + number_of_symbols
            number_of_symbols_true_total = number_of_symbols_true_total + number_of_symbols_true
            if np.abs(number_of_symbols - number_of_symbols_true) <= 1:
                number_of_images_true += 1

percentage_images = number_of_images_true*100 / number_of_images
percentage_symbols = number_of_symbols_true_total*100 / number_of_symbols_total
print("Images = ", number_of_images, number_of_images_true)
print("Symbols = ", number_of_symbols_total, number_of_symbols_true_total)
print("Percentages = ", percentage_images, percentage_symbols)
cv2.waitKey(0)
