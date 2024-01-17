from tqdm import tqdm
import os
import numpy as np
import torch
import shutil

import time

from utils.general import non_max_suppression, xyxy2xywh
from detect import detect_function
from trees_detection_utils.dataset_handler import crop_images
from trees_detection_utils.utils import visualize_bbs



# Global default variables
CROP_SIZE, STEP = 640, 320



def bbs_intersect_ratio(box1, box2):
    """Return the intersection ratio among all possible couples of given bounding boxes.

    Given two bounding boxes $b_1$ and $b_2$, the intersection ratio is the percentage of the area of the 
    smaller box which belongs to the intersection between the two boxes. This intersection ratio is computed
    between all possible couples of boxes among `box1` and `box2`.

    The code is very similar to the code of the function `box_iou` in `yolov7/utils/general.py`.

    Parameters
    ----------
    box1 : torch.Tensor[N, 4]
        First group of $N$ bounding boxes
    box2 : torch.Tensor[M, 4]
        Second group of $M$ bounding boxes

    Returns
    ----------
    intersect_ratios : torch.Tensor[N, M]
        Matrix containing the intersection ratio between all possible couples of bounding boxes.
    rows_min_area_flag : torch.Tensor[N, M]
        Boolean matrix containing, for each couple of boxes, the information whether the smaller box 
        is the first (i.e. row) or second (i.e. column).
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area1 = np.reshape(area1, (area1.shape[0], 1))
    area2 = box_area(box2.T)
    area2 = np.reshape(area2, (area2.shape[0], 1))
    #print(area1.shape)
    #print(area2.shape)

    area1 = np.tile(area1, (1, area1.shape[0]))
    area2 = np.tile(area2, (1, area2.shape[0]))
    #print(area1.shape)
    #print(area2.shape)

    rows_min_area_flag = (area1<area2)
    #print(f'flag shape: {flag.shape}')

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)

    intersect_ratios = np.where(rows_min_area_flag, inter/area1, inter/area2)

    #inter[flag] /= area1
    #inter[~flag] /= area2

    #print(f'inter shape: {inter.shape}')

    return intersect_ratios, rows_min_area_flag



def unify_bbs(predicted_bbs_folder_path, crop_size=CROP_SIZE, output_file_path='./boxes.txt', 
              conf_thres=0.25, iou_nms_thres=0.45, intersect_thres=0.75, verbose=True, verbose_debug=False):
  """Given the bounding boxes computed by the model on each crop, it unifies the boxes for the single full
  image.

  This process involves the following steps.
  1. The bounding boxes are unified and they are transformed from the crop reference system to the same 
     reference system of the full image.
  2. Confidence thresholding and NMS (i.e. non-maxima suppression) are applied. Confidence thresholding 
     consists in removing the bounding boxes with confidence below `conf_thres`, while NMS consists in
     unifying the bounding boxes whose iou (i.e. intersection over union) goes above `iou_nms_thres` (starting
     from the bounding box with higher confidence).
  3. The bounding boxes which are enclosed in others are removed. Basically, the bounding boxes whose 
     intersection ratio with another box goes above `intersect_thres` are removed.

  Two very important parameters: `conf_thres` and `iou_nms_thres`.

  The bounding boxes predicted on each crop are supposed to be stored in `.txt` files. Basically, there is
  a folder containing the predicted bounding boxes info on each crop. In each `.txt` file, each line 
  represents a bounding box, with structure 
            `'{x_center/crop_size} {y_center/crop_size} {w/crop_size} {h/crop_size} {confidence}\n'`
  
  The result of calling this function is a single `.txt` file, containing all unified boxes. Each line 
  represents a bounding box, with structure 
            `'{x_center} {y_center} {w} {h} {confidence}\n'`
            
  Parameters
  ----------
  predicted_bbs_folder_path : str
      Path to the folder into which the `.txt` files containing the predicted bounding boxes info 
      on each crop are put.
  crop_size : int
      Height and width of the crops, i.e. C
  output_file_path : str, optional
      Path into which storing the output `.txt` file, by default './boxes.txt'
  conf_thres : float, optional
      Confidence threshold, by default 0.25
  iou_nms_thres : float, optional
      Iou NMS threshold, by default 0.45
  intersect_thres : float, optional
      Threshold for the intersection ratio, by default 0.75
  verbose : bool, optional
      Whether to be verbose or not, by default True. If verbose, a progress bar is printed.
  verbose_debug : bool, optional
      Whether to print useful debug info or not, by default False
  """  

  # JOINING THE BOUNDING BOXES AND TRANSFORM REFERENCE SYSTEM
  lines = []
  txt_files_names = os.listdir(predicted_bbs_folder_path)
  loop = tqdm(txt_files_names) if verbose else txt_files_names
  for txt_file_name in loop:
    crop_y, crop_x = list(map(int, txt_file_name.split('.')[0].split('_')[1:]))
    txt_file_full_path = os.path.join(predicted_bbs_folder_path, txt_file_name)
    with open(txt_file_full_path) as file:
      curr_txt_lines = [list(map(float, line.rstrip().split()[1:])) for line in file]
    #lines.extend([[line] for line in curr_txt_lines])
    for line in curr_txt_lines:
      x, y, w, h, conf = line
      x = x*crop_size+crop_x
      y = y*crop_size+crop_y
      w = w*crop_size
      h = h*crop_size
      lines.append([x, y, w, h, conf, 1])  # The last value is the probability of belonging to the first 
                                           # class. In our case, since there is just one class which is 'tree',
                                           # this probability is always 1 
  input_nms = np.array(lines)
  #print()
  if verbose_debug:
    print(f'Number of bbs after joining:, {input_nms.shape[0]}')
    #print(input_nms.shape)
    #print(input_nms[0])

  # Sort the bounding boxes, because the NMS algorithm suppose the boxes to be sorted
  sorted_indices = np.argsort(input_nms[:, 4])
  input_nms = input_nms[sorted_indices]

  input_nms = np.expand_dims(input_nms, axis=0)
  input_nms = torch.Tensor(input_nms)

  #print(conf_thres, iou_nms_thres )
  # CONFIDENCE THRESHOLDING AND NMS
  output_nms = non_max_suppression(input_nms, conf_thres, iou_nms_thres)[0]
  output_nms = np.array(output_nms)
  if verbose_debug:
    print(f'Number of bbs after NMS:, {output_nms.shape[0]}')
    #print(output_nms.shape)
    #print(output_nms[0])

  # REMOVAL OF THE ENCLOSED BOXES, based on the intersection ratio
  input_box_inter_perc = torch.Tensor(np.array(output_nms[:, :4]))
  intersect_ratios, rows_min_area_flag = bbs_intersect_ratio(input_box_inter_perc, input_box_inter_perc)
  flag_perc = intersect_ratios>intersect_thres
  flag_perc[list(range(flag_perc.shape[0])),list(range(flag_perc.shape[0]))] = False
  #print(f'flag_perc shape: {flag_perc.shape}')
  #print(f'sum: {flag_perc.sum()}')
  rows_to_remove = np.sum(np.logical_and(flag_perc, rows_min_area_flag), axis=0)>0#.astype(bool)
  #print(f'rows_to_remove shape: {rows_to_remove.shape}')
  columns_to_remove = np.sum(np.logical_and(flag_perc, ~rows_min_area_flag), axis=1)>0#.astype(bool)
  #print(f'columns_to_remove shape: {columns_to_remove.shape}')
  to_remove = np.logical_or(rows_to_remove, columns_to_remove)
  #print(f'to_remove shape: {to_remove.shape}')
  #print(f'sum: {to_remove.sum()}')

  output_nms = output_nms[~to_remove]
  if verbose_debug:
    print(f'Number of bbs after enclosed bbs filtering:, {output_nms.shape[0]}')
    #print(f'output_nms shape: {output_nms.shape}')

  output_nms = xyxy2xywh(output_nms)

  new_lines = [' '.join(map(str, list(xywh)[:-1])) + '\n' for xywh in output_nms]

  #new_lines = list(output_nms)
  with open(output_file_path, "w") as output_file:
    output_file.writelines(new_lines)

  #print(f'Text file saved in {output_file_path}')



def predict_on_img(path_to_weigths, image_path, infrared=True, output_bbs_path='./boxes.txt', 
                   crop_size=CROP_SIZE, step=STEP, conf_thres=0.25, iou_nms_thres=0.45, intersect_thres=0.75, 
                   verbose=False, save_fig_pred_bbs=False, fig_path='./full_figure.jpeg'):
  """Run trees detection inference on one full image 

  The result of calling this function is a single `.txt` file, containing all unified boxes. Each line 
  represents a bounding box, with structure 
               `'{x_center} {y_center} {w} {h} {confidence}\n'`

  This process consists in the following steps.
  1. Crop the given image.
  2. Apply on each crop the model, obtaining the bounding boxes on each crop.
  3. Unify all these bounding boxes for the full single image. This involves multiple steps as well.
      3.1 The bounding boxes are unified and they are transformed from the crop reference system to the 
          same reference system of the full image.
      3.2 Confidence thresholding and NMS (i.e. non-maxima suppression) are applied. Confidence 
          thresholding consists in removing the bounding boxes with confidence below `conf_thres`, while NMS 
          consists in unifying the bounding boxes whose iou (i.e. intersection over union) goes above 
          `iou_nms_thres` (starting from the bounding box with higher confidence).
      3.3 The bounding boxes which are enclosed in others are removed. Basically, the bounding boxes whose
          intersection ratio with another box goes above `intersect_thres` are removed.
     Two very important parameters: `conf_thres` and `iou_nms_thres`.

  Parameters
  ----------
  path_to_weigths : str
      Path to the weigths of the trained yolov7 model
  image_path : str
      Path to the image on which run the inference, typically a `.tif` image
  infrared : bool, optional
      Whether to use also the infrared or to use only the rgb channels, by default True
  output_bbs_path : str, optional
      Path to the `.txt` file into which storing the result, by default './boxes.txt'
  crop_size : int
      Height and width of the crops, i.e. C
  step : int
      Step used for generating the crops, i.e. the stride
  conf_thres : float, optional
      Confidence threshold, by default 0.25
  iou_nms_thres : float, optional
      Iou NMS threshold, by default 0.45
  intersect_thres : float, optional
      Threshold for the intersection ratio, by default 0.75
  verbose : bool, optional
      Whether to be fully verbose or not, by default False. If not fully verbose, only a single progress bar is shown. If 
      fully verbose, each step of the process is shown indipendently with a progress bar.
  save_fig_pred_bbs : bool, optional
      Whether to save a figure with the image and the predicted bounding boxes, by default False
  fig_path : str
      Path into which storing the figure
  """  

  try:
    # Crop image
    if verbose:
        print('CROP IMAGE ...')
    crops_folder_path = './tmp_inference_images'
    if os.path.exists(crops_folder_path):
        shutil.rmtree(crops_folder_path)
    crop_images(image_path, crop_size=crop_size, step=step, folder_path=crops_folder_path, 
                infrared=infrared, verbose=verbose)
    n_crops = len(os.listdir(crops_folder_path))
    if verbose:
        print('FINISH CROP IMAGE')
        print()

    # Run model on each crop
    if verbose:
        print('RUN MODEL ON PATCHES ...')
    project_name = f'inference_{conf_thres}_{iou_nms_thres}'.replace('.', '')
    if os.path.exists(f'runs/detect/{project_name}'):
        shutil.rmtree(f'runs/detect/{project_name}')
    rel_path_to_weigths = os.path.relpath(path_to_weigths)
    detect_function(rgb_only=not infrared, weights=rel_path_to_weigths, conf_thres=conf_thres, iou_thres=iou_nms_thres, source=crops_folder_path, name=project_name, 
                    save_txt=True, save_conf=True)
    if verbose:
        print('FINISH RUN MODEL ON PATCHES')
        print()

    # Unify crops bbs
    if verbose:
        print('UNIFY CROPS BBS ...')
    predicted_bbs_folder_path = f'runs/detect/{project_name}/labels'
    output_file_path = output_bbs_path #f'/content/yolov7/runs/detect/{project_name}/boxes.txt'
    unify_bbs(predicted_bbs_folder_path, crop_size=crop_size, output_file_path=output_file_path, conf_thres=conf_thres,
                iou_nms_thres=iou_nms_thres, intersect_thres=intersect_thres, verbose=verbose)
    if verbose:
        print('FINISH UNIFY CROPS')
        print()

    if save_fig_pred_bbs:
        if verbose:
            print('SAVING IMAGE ...')
        visualize_bbs(image_path, bbs_txt_path=output_file_path, predicted=True, show=False, 
                    fig_path=fig_path, verbose=verbose)
        if verbose:
            print('FINISH SAVING IMAGE')
            
  finally:
    #time.sleep(1)
    # Delete tmp folders
    if os.path.exists(crops_folder_path):
        shutil.rmtree(crops_folder_path)
    if os.path.exists(os.path.join('runs', 'detect')):
        shutil.rmtree(os.path.join('runs', 'detect'))
    if os.path.exists('traced_model.pt'):
        os.remove('traced_model.pt')