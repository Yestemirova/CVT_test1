# Test task for CVT:
import cv2
import numpy as np
import os
import glob
import string
import fnmatch

def segment(original_image, mask_image, comparing_image):
    # Showing Original Images:
    # cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    # cv2.imshow('Original Image', original_image)

    # 1. Binarized Image:
    ret, th1 = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)
    # Showing Binarized Images:
    # cv2.namedWindow('Global Thresholding (v = 127)', cv2.WINDOW_NORMAL)
    # cv2.imshow('Global Thresholding (v = 127)', th1)

    # 2. Contour Image:
    im2, contours, hierarchy = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont = cv2.drawContours(th1, contours, 0, (0, 255, 0), 3)
    # Contour Image:
    # cv2.namedWindow('Contours', cv2.WINDOW_NORMAL)
    # cv2.imshow('Contours', cont)

    # 3. Bounding Box:
    try:
        hierarchy = hierarchy[0]
    except:
        hierarchy = []

    min_y, min_x = th1.shape
    max_y = max_x = 0

    # computes the bounding box for the contour, and draws it on the frame,
    for contour, hier in zip(contours, hierarchy):
        (x, y, w, h) = cv2.boundingRect(contour)
        min_x, max_x = min(x, min_x), max(x + w, max_x)
        min_y, max_y = min(y, min_y), max(y + h, max_y)
        if w > 80 and h > 80:
            cv2.rectangle(th1, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if max_x - min_x > 0 and max_y - min_y > 0:
        rectangle = cv2.rectangle(th1, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
        # cv2.namedWindow('Rectangle', cv2.WINDOW_NORMAL)
        # cv2.imshow('Rectangle', rectangle)

        # 4. Crop the rectangle from Original Image:
        crop = original_image[min_y:max_y, min_x:max_x]
        comparing_crop = comparing_image[min_y:max_y, min_x:max_x]

        # cv2.namedWindow('Cropped', cv2.WINDOW_NORMAL)
        # cv2.imshow('Cropped', crop)
        cv2.namedWindow('Comparing Cropped', cv2.WINDOW_NORMAL)
        cv2.imshow('Comparing Cropped', comparing_crop)

    # 5. adaptiveThreshold of Cropped Image:
    crop_adaptive = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                          cv2.THRESH_BINARY, 11, 2)

    cv2.namedWindow('Adaptive Mean Thresholding of Cropped Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Adaptive Mean Thresholding of Cropped Image', crop_adaptive)

    # 6. Finding Vertical Projections:
    height, width = crop_adaptive.shape
    height_value = np.zeros(width)
    comparing_height, comparing_width = comparing_crop.shape
    number_of_segments_in_comparing_image = number_of_segments_in_cropped_image = 0
    bool_comparing = bool_segment = False

    for w in range(0, width, 1):
        for h in range(height):
            if crop_adaptive[h, w] == 255:
                height_value[w] += 1
            if h == height / 2 and bool_comparing is False and comparing_crop[h, w] == 0:
                number_of_segments_in_comparing_image += 1
                bool_comparing = True
            if h == height / 2 and bool_comparing is True and comparing_crop[h, w] == 255:
                bool_comparing = False

    # 9. Checking with comparison image:
    V_m = max(height_value)
    V_a = round(np.average(height_value))
    V_b = (2 * V_a) - V_m
    # print(V_m, V_a, V_b)

    for w in range(width):
        if height_value[w] < (0.86 * V_m):
            bool_segment = False
        else:
            lines = cv2.line(crop, (w, 0), (w, height), (0, 0, 255), 1)
            cv2.namedWindow('Separating Lines', cv2.WINDOW_NORMAL)
            cv2.imshow('Separating Lines', lines)
            # cv2.imwrite(os.path.join(out_path, 'S.jpg'), lines)
            if bool_segment is False and comparing_crop[comparing_height / 2, w] == 0:
                number_of_segments_in_cropped_image += 1
                bool_segment = True

    print(number_of_segments_in_cropped_image, number_of_segments_in_comparing_image)

    if np.abs(number_of_segments_in_cropped_image - number_of_segments_in_comparing_image) <= 1:
        return True
    else:
        return False

original_image = None
mask_image = None
comparing_image = None
final_result = False
out_path = "/Dataset/Segmented"
filenames = [img for img in glob.glob("Dataset/Dirty LPs2/*.jpg")]
filenames.sort()
name = []
number_of_images = 0
number_of_segment_true = 0

for file in filenames:
    # print ("1. File: ", filenames)
    original_image = None
    mask_image = None
    comparing_image = None
    base = os.path.basename(file)
    name = os.path.splitext(base)[0]
    if name.isdigit():
        number_of_images += 1
        # print("Start: ", name)
        original_image = cv2.imread(file, 0)
        for file1 in filenames:
            base1 = os.path.basename(file1)
            name1 = os.path.splitext(base1)[0]
            # print("For: ", name, "File1 = ", name1)
            if fnmatch.fnmatch(name1, name + "C"):
                # print("If 1: ", name, name1)
                comparing_image = cv2.imread(file1, 0)
            elif fnmatch.fnmatch(name1, name + "M"):
                # print("If 2: ", name, name1)
                mask_image = cv2.imread(file1, 0)
        # print(original_image, mask_image, comparing_image)
        if comparing_image is not None and mask_image is not None:
            # print("If 3: ", name)
            final_result = segment(original_image, mask_image, comparing_image)
            if final_result is True:
                # print("True:", final_result)
                number_of_segment_true += 1

print ("End=", number_of_images, number_of_segment_true)
cv2.waitKey(0)