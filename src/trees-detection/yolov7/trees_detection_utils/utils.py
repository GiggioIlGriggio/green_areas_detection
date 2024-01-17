import numpy as np 
from PIL import Image 
from tqdm import tqdm 
import cv2 
import matplotlib.pyplot as plt

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


def visualize_bbs(image_path, bbs_txt_path, predicted=True, fig_path='./full_figure.jpeg', show=False,
                  verbose=True):
  """Make a figure with the given image and the given bounding boxes.

  The bounding boxes are stored into a `.txt` file. Each line represents a bounding box, with structure 
               `'{x_center} {y_center} {w} {h}\n'`

  If `predicted` is True, then it is assumed that there is also the confidence score for each bounding box.
               `'{x_center} {y_center} {w} {h} {confidence}\n'`

  Parameters
  ----------
  image_path : str
      Path to the image
  bbs_txt_path : _type_
      Path to the bounding boxes `.txt` file
  predicted : bool, optional
      Whether the bounding boxes are being computed by the model or not, by default True
  fig_path : str, optional
      Path to where the figure is saved, by default './full_figure.jpeg'
  show : bool, optional
      Whether to also show the plot in addition to save it, by default False
  verbose : bool, optional 
      Whether to be verbose or not, by default True. If verbose, a progress bar is printed.
  """  

  image = np.array(Image.open(image_path))[:, :, :3].astype(np.uint8)
  #fig, ax = plt.subplots(figsize=(40,40))
  #ax.imshow(image)

  with open(bbs_txt_path) as file:
      boxes_lines = [list(map(float, line.rstrip().split())) for line in file]

  loop = tqdm(boxes_lines) if verbose else boxes_lines
  for line in loop:
      if predicted:
        x, y, w, h, c = line
      else:
        x, y, w, h = line
      """rect = patches.Rectangle((x-w//2, y-h//2), w, h, linewidth=1, edgecolor='purple', facecolor='none')
      # Add the patch to the Axes
      ax.add_patch(rect)"""
      pt1 = (int(x-w//2),int(y-h//2))
      pt2 = (int(x+w//2),int(y+h//2))
      color = (128,0,128)
      thickness = 2
      image = cv2.rectangle(image, pt1, pt2, color, thickness)
      if predicted:
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (128,0,128)  # Purple
        thickness = 1
        image = cv2.putText(image, "{:.2f}".format(c), pt1, font, fontScale, color, thickness, cv2.LINE_AA)

  """extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
  fig.savefig('full_figure.jpeg', bbox_inches='tight')
  print('SAVED')
  #plt.show()
  plt.close()"""
  cv2.imwrite(fig_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
  print(f'Image saved in {fig_path}')
  if show:
    plt.figure(figsize=(20,20))
    plt.imshow(image)