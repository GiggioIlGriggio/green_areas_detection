import os 
from PIL import Image
import numpy as np

from utils.utils import get_crop_index

def load_images_masks(dataset_folder, samples_names, infrared=True):
  """Load the dataset images and the masks

  Parameters
  ----------
  dataset_folder : str
  samples_names : list of str
      List of the names of the samples to use

  Returns
  -------
  images : np.ndarray
    Array of shape (N, H, W, 3), where N is the number of images and (H,W) the
    resolution
  masks : np.ndarray
    Array of shape (N, H, W, 1), where N is the number of images and (H,W) the
    resolution
  """  
  images_list = []
  masks_list = []

  for sample_name in samples_names:
      sample_folder_path = os.path.join(dataset_folder, sample_name)
      #print(sample_folder_path)
      img_path = os.path.join(sample_folder_path, f'{sample_name}.tif')
      #print(img_path)
      mask_path = os.path.join(sample_folder_path, f'{sample_name}_mask.png')
      img = np.array(Image.open(img_path))
      if not infrared:
        img = img[:, :, :3] 
      mask = Image.open(mask_path)
      #print(mask.shape)
      mask = mask.convert("L")
      threshold = 128  # Adjust this threshold value as needed
      mask = mask.point(lambda p: p > threshold and 255)
      mask = np.array(mask)/255
      mask = mask[..., np.newaxis]
      """print(mask.shape)
      mask = np.concatenate([mask, 1-mask], axis=-1)"""
      #print(mask.shape)
      images_list.append(img)
      masks_list.append(mask)

  images = np.array(images_list)
  masks = np.array(masks_list, dtype= np.uint8)

  return images, masks


def crop_images_masks(images, masks, crop_size, step):
  """Crop the given images and masks

  Parameters
  ----------
  images : np.ndarray
    Array of shape (N, H, W, 3), where N is the number of images and (H,W) the
    resolution
  masks : np.ndarray
    Array of shape (N, H, W, 1), where N is the number of images and (H,W) the
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
  cropped_masks : np.ndarray
    Array of shape (M, C, C, 1), where M is the overall number of crops and 
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

  cropped_masks = np.array([masks[:, crop_indices[i][0]:crop_indices[i][0]+crop_size, crop_indices[i][1]:crop_indices[i][1]+crop_size, :] for i in range(len(crop_indices))])
  cropped_masks = np.reshape(cropped_masks, (-1, crop_size, crop_size, 1))

  return cropped_images, cropped_masks