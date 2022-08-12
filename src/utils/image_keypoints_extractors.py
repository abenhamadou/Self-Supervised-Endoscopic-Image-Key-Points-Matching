import cv2
from scipy.spatial import distance
from scipy.ndimage import zoom
import random
import numpy as np


def enhance_image(img):
    crop_img = img[70 : int(img.shape[0]) - 70, 50 : int(img.shape[1]) - 50]
    gray2 = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # clahe = cv.createCLAHE(clipLimit=5)
    gray2 = cv2.blur(gray2, (5, 5))
    # gray2 = cv.medianBlur(gray2 , 3)
    # image_enhanced = clahe.apply(crop_img) + 30
    # gray2= cv.GaussianBlur(gray2, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=5)
    image_enhanced = clahe.apply(gray2)
    # image_enhanced = cv.equalizeHist(gray2)
    return image_enhanced


def zoom_coordinates(img, x, y, zoom_factor, **kwargs):
    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2
        y1 = (y + ((h - zh) // 2)) * zoom_factor
        x1 = (x + ((w - zw) // 2)) * zoom_factor
        # Zero-padding
        out = np.zeros_like(img)
        out[top : top + zh, left : left + zw] = zoom(img, zoom_tuple, **kwargs)
        y1 = np.int32(y1)
        x1 = np.int32(x1)

    # Zooming in
    elif zoom_factor >= 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2
        y1 = (y - ((h - zh) // 2)) * zoom_factor
        x1 = (x - ((w - zw) // 2)) * zoom_factor
        out = zoom(img[top : top + zh, left : left + zw], zoom_tuple, **kwargs)
        y1 = np.int32(y1)
        x1 = np.int32(x1)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        # trim_top = ((out.shape[0] - h) // 2)
        # trim_left = ((out.shape[1] - w) // 2)
        # out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return x1, y1


def clipped_zoom(img, zoom_factor, **kwargs):
    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2
        # Zero-padding
        out = np.zeros_like(img)
        out[top : top + zh, left : left + zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor >= 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top : top + zh, left : left + zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = (out.shape[0] - h) // 2
        trim_left = (out.shape[1] - w) // 2
        out = out[trim_top : trim_top + h, trim_left : trim_left + w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out


def extract_image_keypoints(image, extractor_id):
    assert extractor_id == "SIFT"
    sift = cv2.xfeatures2d.SIFT_create(400)
    keypoints, features = sift.detectAndCompute(image, None)
    return keypoints, features
