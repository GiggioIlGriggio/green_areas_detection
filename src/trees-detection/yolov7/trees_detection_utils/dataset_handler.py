import os

from PIL import Image
import numpy as np
from tqdm import tqdm

import shapefile

from trees_detection_utils.PyMask import tfw
from trees_detection_utils.utils import get_crop_index



############################ CROP IMAGES AND BOUNDING BOXES
def _assign_bbs_to_crops(bbs_shp_path, image_path, crop_size, step):
  """Assign the given bounding boxes into the crops.

  A bounding box is assigned to a crop iif it is fully enclosed into it.

  Parameters
  ----------
  bbs_shp_path : str
      Path to the shp file containing the bounding boxes info
  image_path : str
      Path to the image
  crop_size : int
      Height and width of the crops, i.e. C
  step : int
      Step used for generating the crops, i.e. the stride

  Returns
  -------
  crops_bbs_dict : dict 
      Dict containing all the crops and the lists of bounding boxes assigned to them.
      Each entry corresponds to a crop. The key is the crop top-left corner coordinates, i.e. `(y, x)`.
      The value is the list of bounding boxes enclosed in that crop. Each bounding box is represented 
      as a string, with structure 
              `'0 {x_center/crop_size} {y_center/crop_size} {w/crop_size} {h/crop_size}\n'`
      where `x_center, y_center` are the coordinates of the bounding box center (w.r.t. the crop reference 
      system), `w, h` are the bounding box dimensions and `crop_size` is the size of the crop.
  
  bbs_assignment_frequency_dict : dict
      Dict containing auxiliary info about this assignment of the boxes to the crops.
      The keys are the number of crops into which each different bounding box is stored, and the values are
      the number of bounding boxes with that number of crops.
  """
  try:
      tf = tfw(image_path)
  except Exception as err:
      print(err.strerror)
      return

  image = np.array(Image.open(image_path))
  H, W = image.shape[:-1]

  try:
      shape = shapefile.Reader(bbs_shp_path)
  except Exception as err:
      print(err.args[0])
      return
  if shape.dbf is None or shape.shp is None:
      print('Errore apertura file ' + bbs_shp_path)
      return

  crop_indices = get_crop_index(crop_size=crop_size, step=step, w=W, h=H)
  crops_bbs_dict = {}
  bbs_assignment_frequency_dict = {}

  crop_indices_y, crop_indices_x = map(list,zip(*crop_indices))
  crop_indices_y = np.array(crop_indices_y)
  crop_indices_x = np.array(crop_indices_x)

  crop_indices = np.array(crop_indices)

  shape_records = shape.shapeRecords()

  for feature in shape_records:
      #print(f"{i}/{shape.numShapes}")
      #feature = shape.shapeRecords()[i]

      x0, y0 = tf.toImg(feature.shape.bbox[0], feature.shape.bbox[3])
      x1, y1 = tf.toImg(feature.shape.bbox[2], feature.shape.bbox[1])

      w = abs(x1-x0)
      h =  abs(y1-y0)

      bool_arr1 = crop_indices_x<=x0
      bool_arr2 = crop_indices_x+crop_size>=x0+w
      bool_arr3 = crop_indices_y<=y0
      bool_arr4 = crop_indices_y+crop_size>=y0+h
      bool_arr5 = np.logical_and(bool_arr1, bool_arr2)
      bool_arr6 = np.logical_and(bool_arr3, bool_arr4)
      bool_arr = np.logical_and(bool_arr5, bool_arr6)

      crops = crop_indices[bool_arr, :]

      count = crops.shape[0]
      for crop_y0, crop_x0 in crops:
        x0_rel = x0 - crop_x0
        y0_rel = y0 - crop_y0

        x_center = x0_rel + w//2
        y_center = y0_rel + h//2

        #print(f'Match {count}: {x_center} {y_center} {w} {h}')

        txt = f'0 {x_center/crop_size} {y_center/crop_size} {w/crop_size} {h/crop_size}\n'
        crop_key = (crop_y0, crop_x0)
        crops_bbs_dict[crop_key] = crops_bbs_dict.get(crop_key, '') + txt

      bbs_assignment_frequency_dict[count] =  bbs_assignment_frequency_dict.get(count, 0) + 1

  sum = 0
  for count in bbs_assignment_frequency_dict.keys():
    sum += bbs_assignment_frequency_dict[count]
  for count in bbs_assignment_frequency_dict.keys():
    bbs_assignment_frequency_dict[count] /= sum
  bbs_assignment_frequency_dict = dict(sorted(bbs_assignment_frequency_dict.items(), key=lambda item: item[0]))
  bbs_assignment_frequency_dict = {key:round(value, 2) for key, value in bbs_assignment_frequency_dict.items()}

  return crops_bbs_dict, bbs_assignment_frequency_dict


def _save_cropped_image_and_bbs(crops_bbs_dict, image_path, crop_size, infrared=True, folder_path='./'):
  """Given an image and the assignment of its bounding boxes to the crops,it saves the crops and the 
  corresponding bounding boxes `.txt` info files.

  The crops are saved into the subfolder `images`, while the bounding boxes `.txt` info files into the 
  `labels` subfolder.

  Each line in each `.txt` file is a bounding box, with structure 
                `'0 {x_center/crop_size} {y_center/crop_size} {w/crop_size} {h/crop_size}\n'`
  (The starting 0 refer to the class, which is 'tree' in our case).

  Parameters
  ----------
  crops_bbs_dict : dict
      Dict containing all the crops and the lists of bounding boxes assigned to them.
  image_path : str
      Path to the image
  crop_size : int
      Height and width of the crops, i.e. C
  infrared : bool, optional
      Whether to use also the infrared or only rgb channels, by default True
  folder_path : str, optional
      Folder into which storing the results, by default './'
  """

  #shutil.rmtree(folder_path)
  os.makedirs(folder_path, exist_ok =True)
  dataset_images_path = os.path.join(folder_path, "images")
  os.makedirs(dataset_images_path, exist_ok =True)
  dataset_labels_path = os.path.join(folder_path, "labels")
  os.makedirs(dataset_labels_path, exist_ok =True)
  image = np.array(Image.open(image_path))
  for crop_coord, txt in tqdm(crops_bbs_dict.items()):
    if infrared:
      crop = image[crop_coord[0]:crop_coord[0]+crop_size, crop_coord[1]:crop_coord[1]+crop_size, :]
    else:
      crop = image[crop_coord[0]:crop_coord[0]+crop_size, crop_coord[1]:crop_coord[1]+crop_size, :3]

    to_save = Image.fromarray(crop)
    img_name = os.path.basename(image_path).split('.')[0]
    name = f'{img_name}_{crop_coord[0]}_{crop_coord[1]}'  # y0 x0
    to_save.save(os.path.join(dataset_images_path, name + '.tif'))

    with open(os.path.join(dataset_labels_path, name + '.txt'), 'w') as file:
      file.write(txt)


def crop_images_and_bbs(image_paths_list, bbs_shp_paths_list, crop_size, step, infrared=True, 
                        folder_path='./train'):
  """Crop the given images and the given bounding boxes.

  Each image is splitted into crops. Moreover, all the bounding boxes inside that image are splitted 
  into crops as well, meaning that they are assigned to the different crops. (A bounding box is assigned 
  to a crop iif it is fully enclosed in that crop).

  Going more in depth, the result of calling this function is the creation of two folders.
  - Folder `images`, into which the crops of all images are stored. The crops are `.tif` images.
  - Folder `labels`, into which the corresponding bounding boxes are stored. For each crop, a corresponding
    `.txt` file is present, containing the bounding boxes enclosed in that crop.

  So, each `.txt` file contains the bounding boxes enclosed in the corresponding crop. Each line represents
  a bounding box, with structure `0 xc yc w h`, where `xc, yc` are the coordinates of the bounding box center
  (w.r.t. the crop reference system, i.e. the top-left cornet of the crop) and `w, h` are the bounding box 
  dimensions. These four quantities are relative w.r.t. the crop, meaning that they have been divided by 
  the crop size. 
            `'0 {x_center/crop_size} {y_center/crop_size} {w/crop_size} {h/crop_size}\n'`
  (The starting 0 refer to the class, which is 'tree' in our case).

  Parameters
  ----------
  image_paths_list : list of str
      List of paths of the images to crop. The images are `.tif` files.
  bbs_shp_paths_list : list of str
      List of paths to the shp files containing the bounding boxes info
  crop_size : int
      Height and width of the crops, i.e. C
  step : int
      Step used for generating the crops, i.e. the stride
  infrared : bool, optional
      Whether to use also the infrared or only rgb channels, by default True
  folder_path : str, optional
      Folder into which storing the results, by default './train'
  """

  # Iterate over all given images
  for image_path, bbs_shp_path in zip(image_paths_list, bbs_shp_paths_list):
    img_name = os.path.basename(image_path).split('_')[0]
    print(f'Processing image {img_name}')
    # Assign bounding boxes to the crops 
    crops_bbs_dict, bbs_assignment_frequency_dict = _assign_bbs_to_crops(bbs_shp_path, image_path, crop_size, step)
    # Save the crops and the corresponding bounding boxes info
    _save_cropped_image_and_bbs(crops_bbs_dict, image_path, crop_size, infrared=infrared, folder_path=folder_path)
    print('BBs frequencies assignment', bbs_assignment_frequency_dict)
    print()



############################ CROP IMAGES
def crop_images(image_path, crop_size, step, infrared=True, folder_path='./images', verbose=True):
    """Crop the given image.

    The crops are saved as `.tif` files, into the specified folder.

    Parameters
    ----------
    image_path : str
        Paths of the image to crop, which is a `.tif` file. 
    crop_size : int
        Height and width of the crops, i.e. C
    step : int
        Step used for generating the crops, i.e. the stride
    infrared : bool, optional
        Whether to use also the infrared or only rgb channels, by default True
    folder_path : str, optional
        Folder into which storing the results, by default './images'
    verbose : bool, optional 
        Whether to be verbose or not, by default True. If verbose, a progress bar is printed.
    """
    #print(image_path)
    os.makedirs(folder_path, exist_ok=True)

    image = np.array(Image.open(image_path))
    H, W = image.shape[:-1]
    #print(H, W)

    crop_indices = get_crop_index(crop_size=crop_size, step=step, w=W, h=H)

    crop_indices = np.array(crop_indices)

    loop = tqdm(crop_indices) if verbose else crop_indices
    for crop_y0, crop_x0 in loop:
      if infrared:
        crop = image[crop_y0:crop_y0+crop_size, crop_x0:crop_x0+crop_size, :]
      else: 
        crop = image[crop_y0:crop_y0+crop_size, crop_x0:crop_x0+crop_size, :3]
      to_save = Image.fromarray(crop)
      img_name = os.path.basename(image_path).split('.')[0]
      name = f'{img_name}_{crop_y0}_{crop_x0}'  # y0 x0
      to_save.save(os.path.join(folder_path, name + '.tif'))



############################ SAVE BOUNDING BOXES AS txt
def save_bbs_as_txt(bbs_shp_path, image_path, output_file_path=None):
  """Save the given `.shp` bounding boxes info file as `.txt` file.

  The `.txt` file has the following format. Each line represents a bounding box, with structure 
  `xc yc w h`, where `xc, yc` are the coordinates of the bounding box center (w.r.t. the crop reference 
  system, i.e. the top-left cornet of the crop) and `w, h` are the bounding box dimensions. 
                            `'{x_center} {y_center} {w} {h}\n'`

  Parameters
  ----------
  bbs_shp_path : str
      Path to the `.shp` file
  image_path : str
      Path to the image, `.tif` file.
  output_file_path : str, optional
      Path of the file into which storing the `.txt` file, by default None.
      If None, `output_file_path` is the same as `image_path`, but with `.txt` extension.
  """
  try:
      tf = tfw(image_path)
  except Exception as err:
      print(err.strerror)
      return

  if output_file_path is None:
    output_file_path = image_path.split('.')[0] + '_boxes.txt'

  image = np.array(Image.open(image_path))
  H, W = image.shape[:-1]

  lines = []

  try:
      shape = shapefile.Reader(bbs_shp_path)
  except Exception as err:
      print(err.args[0])
      return
  if shape.dbf is None or shape.shp is None:
      print('Errore apertura file ' + bbs_shp_path)
      return

  shape_records = shape.shapeRecords()

  for feature in shape_records:
      #print(f"{i}/{shape.numShapes}")
      #feature = shape.shapeRecords()[i]

      x0, y0 = tf.toImg(feature.shape.bbox[0], feature.shape.bbox[3])
      x1, y1 = tf.toImg(feature.shape.bbox[2], feature.shape.bbox[1])

      w = abs(x1-x0)
      h =  abs(y1-y0)

      x_center = x0 + w//2
      y_center = y0 + h//2

      #print(f'Match {count}: {x_center} {y_center} {w} {h}')

      txt = f'{x_center} {y_center} {w} {h}\n'
      lines.append(txt)

  with open(output_file_path, "w") as output_file:
    output_file.writelines(lines)