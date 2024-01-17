""" 
The starting point is the `utils` file taken from the `autocartography` project 
for roof detection. Then, it has been modified and improved.

It contains several utility functions. The three most important are the following.
- `get_crop_index`, for computing the crops indices.
- `predict_on_img`, for computing binary predictions on a full image.
- `get_polylines`, for computing the polylines from the binary mask.
"""

from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import cv2
from shapely.geometry import Polygon

#from utils.dataset_handler import crop_images



########################### DEFAULT VALUES
CROP_SIZE = 1024
STEP = 512
SIGMA = 512  # stddev of gaussian weighting
N_CLASSES = 1

def get_palette():
    palette = [
        0, 0, 0,  # black
        255, 255, 255  # white
    ]

    # Zero-pad the palette to 256 RGB colours, i.e. 768 values
    palette += (768 - len(palette)) * [0]
    return palette



########################### GET CROP INDEX
def get_crop_index(crop_size, step, w, h):
    """Return the crops indices. 

    In particular, it returns the indices of the top-left corner of the crops.

    Parameters
    ----------
    crop_size : int
        Height and width of the crops, i.e. C
    step : int
        Step used for generating the crops, i.e. the stride
    w : int
        Width of the original image
    h : int
        Height of the original image

    Returns
    -------
    crop_indices : list of list
        List of couples (y_i, x_i), representing the coordinates of the top-left
        corner of each crop.
        So, the length of `crop_indices` is M, which is the total number of crops.
    """
    y_cur = 0  # upper left
    crop_indices = []

    while y_cur + crop_size < h:  # top-down

        x_cur = 0
        crop_indices += [[y_cur, x_cur]]

        while x_cur + crop_size < w:  # left to right
            if x_cur + step + crop_size < w:
                x_cur += step
            else:
                x_cur = w - crop_size
            crop_indices += [[y_cur, x_cur]]

        if y_cur + step + crop_size < h:
            y_cur += step
        else:  # y of last row
            y_cur = h - crop_size
            x_cur = 0
            crop_indices += [[y_cur, x_cur]]

        # last row
        while x_cur + crop_size < w:
            if x_cur + step + crop_size < w:
                x_cur += step
            else:
                x_cur = w - crop_size
            crop_indices += [[y_cur, x_cur]]
    return crop_indices


def crop_images(images, crop_size, step):
  """Crop the given images

  Parameters
  ----------
  images : np.ndarray
    Array of shape (N, H, W, 3), where N is the number of images and (H,W) the
    resolution
  crop_size : int
      Height and width of the crops, i.e. C
  step : int
      Step used for generating the crops, i.e. the stride

  Returns
  -------
  cropped_images : np.ndarray
    Array of shape (M, C, C, 3), where M is the overall number of crops and 
    (C,C) the resolution of each crop
  """
  h, w = images[0].shape[:-1]

  # Indices of the top-left crops corners
  crop_indices = get_crop_index(crop_size=crop_size, step=step, w=w, h=h)

  cropped_images = np.array([images[:, crop_indices[i][0]:crop_indices[i][0]+crop_size, crop_indices[i][1]:crop_indices[i][1]+crop_size, :] for i in range(len(crop_indices))])
  num_channels = cropped_images.shape[-1]
  cropped_images = np.reshape(cropped_images, (-1, crop_size, crop_size, num_channels))
  #infrared = cropped_images[..., -1]
  #cropped_images = cropped_images[..., :-1]  # TODO: removed infrared

  return cropped_images



########################### PREDICT ON IMG
def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

def gaussian_square(size, sigma):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

    kernel_2D *= 1.0 / kernel_2D.max()

    return kernel_2D

def crop_map(img, index):
    x0 = tf.constant(np.array(0, dtype=np.int32).reshape((1,)))
    b = tf.concat([index, x0], axis=0)
    n_channels = img.shape[2]
    crop = tf.slice(img, begin=b, size=tf.constant([CROP_SIZE, CROP_SIZE, n_channels]))
    #crop = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1. / 127.5, offset=-1)(crop)
    return crop, index


def predict_on_img(model, img, batch_size, crop_size=CROP_SIZE, step=STEP, 
                   use_gaussian=False):
    """Compute the binary predictions on the full input image, returning a full 
    predicted mask.

    Basically, the image is splitted into small crops and the model is applied on
    each crop, returning the crop binary predictions. Then, all these crops masks 
    are merged together into a single full mask.

    The merging of the crops binary predictions is important, since the crops can 
    be overlapping. In other words, a pixel can receive more binary predictions.
    The final binary prediction on a pixel is simply computed as the average of 
    such several predictions.

    Optionally, the binary predictions computed on a crop can be rescaled by means
    of a gaussian filter, such that values computed far from the crop center are
    weakened (i.e. smaller values). The idea is that predictions near the crop
    center are reliable, and so they are not smoothened; but the far we go from
    the center, the less reliable are the values. However, a problem of this
    approach is that the predictions lose the semantic of beeing probabilities
    in the range [0,1]. For solving this, we could try to do a weighted average
    instead of a normal average. TODO

    Parameters
    ----------
    model : tensorflow.keras.Model
        Trained model
    img : np.ndarray [H, W, 3]
        Full input image
    batch_size : int
        Number of crops processed in paralell.
    crop_size : int
        Height and width of the crops, i.e. C
    step : int
        Step used for generating the crops, i.e. the stride
    use_gaussian : bool, optional
        Whether to use the gaussian rescaling or not, by default False

    Returns
    -------
    preds : np.ndarray [H, W, 1]
        Full predicted binary mask
    """

    # gaussian weight
    # Gaussian filter
    gaussian_weight = gaussian_square(CROP_SIZE, SIGMA)
    gaussian_weight = np.repeat(gaussian_weight[np.newaxis, :], batch_size, axis=0)
    gaussian_weight = np.repeat(gaussian_weight[:, :, :, np.newaxis], N_CLASSES, axis=-1)

    h, w = img.shape[:2]
    # Array [H, W, 1] in which we accumulate the per-pixel predictions
    preds = np.zeros((h, w, N_CLASSES), dtype='float16')  # combined predictions
    # Array [H, W] in which we accumulate the per-pixel occurances, i.e. the 
    # number of predictions done for each pixel
    occs = np.zeros((h, w), dtype='uint8')  # pixel-wise prediction count

    # Indices of the top-left crops corners
    crop_indices = get_crop_index(crop_size=crop_size, step=step, w=w, h=h)

    # Iterate over batches of crops
    images = np.expand_dims(img, axis=0)
    for i in tqdm(range(0, len(crop_indices), batch_size)):
        # Compute the batch of crops `crops`
        crop_indices_curr = crop_indices[i:i+batch_size]
        crops = np.array([images[:, crop_indices_curr[j][0]:crop_indices_curr[j][0]+crop_size, crop_indices_curr[j][1]:crop_indices_curr[j][1]+crop_size, :] for j in range(batch_size)])
        num_channels = crops.shape[-1]
        crops = np.reshape(crops, (-1, crop_size, crop_size, num_channels))

        # Binary predictions on that crops
        crop_preds = model.predict(crops, verbose=0)

        if use_gaussian: # Rescale using the gaussian filter
            crop_preds *= gaussian_weight[:crop_preds.shape[0]]

        # Update `preds` and `occs`
        for j in range(batch_size):
            y_cur, x_cur = crop_indices[i+j]
            preds[y_cur:y_cur + crop_size, x_cur:x_cur + crop_size] += crop_preds[j]
            occs[y_cur:y_cur + crop_size, x_cur:x_cur + crop_size] += 1  # update prediction count


    # Let's compute the average
    occs = occs[..., np.newaxis]
    preds /= occs

    del occs

    return preds



########################### GET POLYLINES
def vectorize(binary_mask: np.ndarray) -> List[List[Tuple[float]]]:
    """
    Opencv based implementation of the vectorization of a binary mask
    :param binary_mask: a raster binary mask where ones indicates the objects.
    :returns a list of polygons. Each polygon is as a list of (x, y) coordinates.
    """

    mask = binary_mask.astype(np.uint8)
    mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    polygons = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE, offset=(-1, -1))
    polygons = polygons[0] if len(polygons) == 2 else polygons[1]
    # polygons = [[tuple(pair) for pair in polygon.flatten()] for polygon in polygons]

    valid_polygons = []
    for p in polygons:
        x, y = p[:, 0, 0], p[:, 0, 1]
        if len(p) >= 3:  # an object must be closed
            p = [(x[i], y[i]) for i in range(len(x))]
            valid_polygons += [p + [p[0]]]  # add last vertex
    return valid_polygons


def get_polylines(binary_preds, tolerance: float, min_area: float):
    """Return the polylines from the thresholded binary mask

    Parameters
    ----------
    binary_preds : np.ndarray [H, W, 1]
        Thresholded binary mask, containing just 0/1
    tolerance : float
    min_area : float

    Returns
    -------
    simplified_polygons : list
        List of polylines
    """
    points = vectorize(binary_preds)

    simplified_polygons = []
    for poly_points in points:
        p = Polygon(poly_points)
        if p.area >= min_area:
            p = p.simplify(tolerance=tolerance)
            simplified_polygons += [list(p.exterior.coords)]
    return simplified_polygons
