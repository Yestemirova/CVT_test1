import cv2
import numpy as np

def segment(original_image, mask_image, comparing_image, name):
    # 1. Binarized Image:
    ret, th1 = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)

    # 2. Contour Image:
    im2, contours, hierarchy = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont = cv2.drawContours(th1, contours, 0, (0, 255, 0), 3)

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

        # 4. Crop the rectangle from Original and Comparing Image:
        crop = original_image[min_y:max_y, min_x:max_x]
        comparing_crop = comparing_image[min_y:max_y, min_x:max_x]

    # 5. adaptiveThreshold of Cropped Image:
    crop_adaptive = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                          cv2.THRESH_BINARY, 11, 2)
    # 6. Finding Vertical Projections:
    height, width = crop_adaptive.shape
    height_value = np.zeros(width)
    comparing_height, comparing_width = comparing_crop.shape
    number_of_segments_in_comparing_image = number_of_segments_in_cropped_image = 0
    bool_comparing = bool_segment = False
    comparing_symbols = np.zeros((comparing_width, 2))
    crop_symbols = np.zeros((width, 2))

    for w in range(0, width, 1):
        for h in range(height):
            if crop_adaptive[h, w] == 255:
                height_value[w] += 1
            if h == height / 2 and bool_comparing is False and comparing_crop[h, w] == 255:
                comparing_symbols[number_of_segments_in_comparing_image][0] = w
                bool_comparing = True
            if h == height / 2 and bool_comparing is True and comparing_crop[h, w] == 0:
                comparing_symbols[number_of_segments_in_comparing_image][1] = w
                number_of_segments_in_comparing_image += 1
                bool_comparing = False
            if h == height / 2 and bool_comparing is True and w == comparing_width-2 and comparing_crop[h, w] != 0:
                comparing_symbols[number_of_segments_in_comparing_image][1] = w
                number_of_segments_in_comparing_image += 1

    # 9. Checking with comparison image:
    V_m = max(height_value)
    i = 0

    for w in range(width):
        if height_value[w] < (0.8 * V_m):
            if bool_segment is False and i == 0:
                crop_symbols[number_of_segments_in_cropped_image][0] = w
                bool_segment = True
                i += 1
            if bool_segment is True:
                i += 1
        else:
            if bool_segment is True and i > 10:
                lines = cv2.line(crop, (w, 0), (w, height), (0, 0, 255), 1)
                cv2.namedWindow('Separating Lines', cv2.WINDOW_NORMAL)
                cv2.imshow('Separating Lines', lines)
                crop_symbols[number_of_segments_in_cropped_image][1] = w-1
                number_of_segments_in_cropped_image += 1
                bool_segment = False
                i = 0
            if bool_segment is False and i < 5:
                i = 0
    cv2.imwrite('Dataset/Segmented/' + name + 'S.jpg', lines)

    number_of_symbols_true = 0
    for n in range(number_of_segments_in_comparing_image):
        if np.abs(crop_symbols[n+1][0] - comparing_symbols[n][0]) < 5 or np.abs(
                crop_symbols[n+1][1] - comparing_symbols[n][1]) < 5:
            number_of_symbols_true += 1
    return number_of_segments_in_comparing_image, number_of_symbols_true
