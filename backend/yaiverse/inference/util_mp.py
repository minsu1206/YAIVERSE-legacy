import os
#import dlib
from PIL import Image
import numpy as np
import scipy
import scipy.ndimage
import cv2
import math


def image_align(image, center, theta):
    height, width = image.shape[:2]
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=theta, scale=1)
    rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
    return rotated_image

def align_face_mp(filepath, lm, set_size, box_list, output_size=1024, transform_size=4096, enable_padding=True):
    """
    :param filepath: str
    :return: PIL Image
    """
    # Calculate auxiliary vectors.
    eye_left = lm[1]
    eye_right = lm[0] 
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_avg = lm[3]
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = np.array([1, 0], dtype=np.float64)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]        
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2
    
    # read image
    img = cv2.imread(filepath)
    img = square_padding(img)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize((set_size, set_size), Image.ANTIALIAS)
    print("img.size in align_face_mp", img.size)

    transform_size = output_size
    enable_padding = True

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        print('consider shrink')
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)))
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # # image align
    degree = (lm[1][1] - lm[0][1])/(lm[1][0] - lm[0][0])
    img = np.asarray(img)
    center_p = (box_list[0] + box_list[2]/2 , box_list[1] + box_list[2]/2)
    img_aligned = image_align(img, center_p,  math.degrees(math.atan(degree)))
    img = Image.fromarray(img_aligned)

    #Transform. --> align
    img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
    if output_size < transform_size:
        print('change size')
        img = img.resize((output_size, output_size), Image.ANTIALIAS)

    
    return img


def square_padding(image):
    h, w = image.shape[:2]
    maxhw = np.max([h, w])
    pad_h = int((maxhw - h) / 2)
    pad_v = int((maxhw - w) / 2)
    return cv2.copyMakeBorder(image, pad_h, pad_h, pad_v, pad_v, cv2.BORDER_CONSTANT)