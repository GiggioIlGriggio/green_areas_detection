import torch
import numpy as np
from PIL import Image 
from tqdm import tqdm 
import cv2
import matplotlib.pyplot as plt

from utils.general import box_iou, xywh2xyxy
from trees_detection_utils.inference import predict_on_img


# Default global values
CROP_SIZE = 640 
STEP = 320



####################### VISUALIZE ERRORS AND MISSINGS
def visualize_errors_and_missings(image_path, gt_boxes_txt_path, pred_boxes_txt_path, iou_eval_thres=0.4, 
                                  show=False, fig_path='./full_figure_with_errors_and_missings.jpeg'):
    """Visualize the errors and the missings of the given predicted bounding boxes w.r.t. the given ground 
    truth bounding boxes.

    The 'errors' are the predicted bounding boxes which do not correspond to any ground truth box, i.e. the
    false positives. The 'mistakes' are the ground truth boxes which are not matched by any predicted box.
    The former are used to visualize the precision of the model, the latter the recall.

    In the figure, the following boxes are made.
        - Blue boxes. The boxes predicted by the model which are errors.
        - Azure boxes. The boxes predicted by the model which are correct, i.e. they match at least one ground 
          truth box.
        - Red boxes. The ground truth boxes which are missings.
        - Pink boxes. The ground truth boxes which are matched by at least one predicted box.
    With the predicted boxes, also the confidence score is shown.
    
    The matching of the predicted boxes with the ground truth boxes is made with the computation of the iou.
    A predicted box matches a ground truth one if their iou is above `iou_eval_thres`. 

    The ground truth boxes are stored in a `.txt` file. Each line represents a bounding box, with 
    structure 
                   `'{x_center} {y_center} {w} {h}\n'`

    The predicted truth boxes are stored in a `.txt` file. Each line represents a bounding box, with 
    structure 
                   `'{x_center} {y_center} {w} {h} {confidence}\n'`

    Parameters
    ----------
    image_path : str
        Path to the image
    gt_boxes_txt_path : str
        Path to the `.txt` file containing the ground truth boxes
    pred_boxes_txt_path : str
        Path to the `.txt` file containing the predicted boxes
    iou_eval_thres : float, optional
        Iou threshold used for matching predictions-ground truths, by default 0.4
    fig_path : str
        Path into which storing the figure
    show : bool, optional
        Whether to also show the figure and not only saving it, by default False
    """

    image = np.array(Image.open(image_path))[:, :, :3]

    with open(gt_boxes_txt_path) as file:
        gt_boxes = np.array([list(map(float, line.rstrip().split())) for line in file])

    with open(pred_boxes_txt_path) as file:
        pred_boxes = np.array([list(map(float, line.rstrip().split())) for line in file])

    gt_pred_ious = box_iou(torch.Tensor(xywh2xyxy(gt_boxes)[:, :4]), torch.Tensor(xywh2xyxy(pred_boxes)[:, :4]))
    gt_pred_ious = np.array(gt_pred_ious)
    #print(gt_pred_ious.shape)

    gt_pred_ious_mask = (gt_pred_ious>=iou_eval_thres).astype(int)

    # Errors
    pred_errors_indices = np.sum(gt_pred_ious_mask, axis=0)==0
    #print(pred_errors_indices.shape)
    #print(pred_errors_indices.sum())
    #pred_errors_boxes = pred_boxes[pred_errors_indices]
    #print(pred_errors_boxes.shape)
    pred_boxes = np.concatenate([pred_boxes, np.reshape(pred_errors_indices, (pred_errors_indices.shape[0],1))], axis=1)

    # Missings
    gt_missings_indices = np.sum(gt_pred_ious_mask, axis=1)==0
    #print(gt_missings_indices.shape)
    #print(gt_missings_indices.sum())
    #gt_missings_boxes = gt_boxes[gt_missings_indices]
    #print(gt_missings_boxes.shape)
    gt_boxes = np.concatenate([gt_boxes, np.reshape(gt_missings_indices, (gt_missings_indices.shape[0],1))], axis=1)

    # Show predicted boxes
    for x, y, w, h, c, b in tqdm(pred_boxes):
        pt1 = (int(x-w//2),int(y-h//2))
        pt2 = (int(x+w//2),int(y+h//2))
        color = (0,0,256) if b else (0, 127, 255) # Blue or Azure
        thickness = 2
        image = cv2.rectangle(image, pt1, pt2, color, thickness)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        #color = (0,256,0) if b else (0, 127, 255) # Blue or Azure
        thickness = 1
        image = cv2.putText(image, "{:.2f}".format(c), pt1, font, fontScale, color, thickness, cv2.LINE_AA)

    # Show ground truth boxes
    for x, y, w, h, b in tqdm(gt_boxes):
        pt1 = (int(x-w//2),int(y-h//2))
        pt2 = (int(x+w//2),int(y+h//2))
        color = (256,0,0) if b else (255,192,203) # Red or Pink
        thickness = 2
        image = cv2.rectangle(image, pt1, pt2, color, thickness)

    cv2.imwrite(fig_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print('Figure saved')
    if show:
        plt.figure(figsize=(20,20))
        plt.imshow(image)



####################### EVALUATE MODEL
# Part of the code is taken from https://github.com/klimaschkas/mapcalc
class _ImageDetection:
    def __init__(self, score, label, boxes, used=False):
        self.boxes = boxes
        self.label = label
        self.score = score
        self.used = used


def _voc_ap(rec, prec):
    """
     Calculate the AP given the recall and precision array
        1st) We compute a version of the measured precision/recall curve with
             precision monotonically decreasing
        2nd) We compute the AP as the area under this curve by numerical integration.

    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """

    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]

    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """

    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def _check_dicts_for_content_and_size(ground_truth_dict: dict, result_dict: dict):
    """

    Checks if the content and the size of the arrays adds up.
    Raises and exception if not, does nothing if everything is ok.

    :param ground_truth_dict: dict with {boxes:, labels:}
    e.g.
    {
    'boxes':
        [[60., 80., 66., 92.],
         [59., 94., 68., 97.],
         [70., 87., 81., 94.],
         [8., 34., 10., 36.]],

    'labels':
        [2, 2, 3, 4]}
    :param result_dict: dict with {boxes:, labels:, scores:}
    e.g.
    {
    'boxes':
        [[57., 87., 66., 94.],
         [58., 94., 68., 95.],
         [70., 88., 81., 93.],
         [10., 37., 17., 40.]],

    'labels':
        [2, 3, 3, 4],

    'scores':
        [0.99056727, 0.98965424, 0.93990153, 0.9157755]}
    :return:
    """
    if 'boxes' not in ground_truth_dict.keys():
        raise ValueError("ground_truth_dict expects the keys 'boxes' and 'labels'.")
    if 'labels' not in ground_truth_dict.keys():
        raise ValueError("ground_truth_dict expects the keys 'boxes' and 'labels'.")
    if 'boxes' not in result_dict.keys():
        raise ValueError("result_dict expects the keys 'boxes' and 'labels' and optionally 'scores'.")
    if 'labels' not in result_dict.keys():
        raise ValueError("result_dict expects the keys 'boxes' and 'labels' and optionally 'scores'.")

    if 'scores' not in result_dict.keys():
        result_dict['scores'] = [1] * len(result_dict['boxes'])

    if len(ground_truth_dict['boxes']) != len(ground_truth_dict['labels']):
        raise ValueError("The number of boxes and labels differ in the ground_truth_dict.")

    if not len(result_dict['boxes']) == len(result_dict['labels']) == len(result_dict['scores']):
        raise ValueError("The number of boxes, labels and scores differ in the result_dict.")


def calculate_map(ground_truth_dict: dict, result_dict: dict, iou_threshold: float):
    """
    mAP@[iou_threshold]

    :param ground_truth_dict: dict with {boxes:, labels:}
    e.g.
    {
    'boxes':
        [[60., 80., 66., 92.],
         [59., 94., 68., 97.],
         [70., 87., 81., 94.],
         [8., 34., 10., 36.]],

    'labels':
        [2, 2, 3, 4]}
    :param result_dict: dict with {boxes:, labels:, scores:}
    e.g.
    {
    'boxes':
        [[57., 87., 66., 94.],
         [58., 94., 68., 95.],
         [70., 88., 81., 93.],
         [10., 37., 17., 40.]],

    'labels':
        [2, 3, 3, 4],

    'scores':
        [0.99056727, 0.98965424, 0.93990153, 0.9157755]}
    :param iou_threshold: minimum iou for which the detection counts as successful
    :return: mean average precision (mAP)
    """

    # checking if the variables have the correct keys

    _check_dicts_for_content_and_size(ground_truth_dict, result_dict)

    occurring_gt_classes = set(ground_truth_dict['labels'])
    unique, counts = np.unique(ground_truth_dict['labels'], return_counts=True)
    ground_truth_counter_per_class = dict(zip(unique, counts))
    count_true_positives = {}
    sum_average_precision = 0

    for class_index, class_name in enumerate(occurring_gt_classes):

        detections_with_certain_class = list()
        for idx in range(len(result_dict['labels'])):
            if result_dict['labels'][idx] == class_name:
                detections_with_certain_class.append(_ImageDetection(score=result_dict['scores'][idx],
                                                                     label=result_dict['labels'][idx],
                                                                     boxes=result_dict['boxes'][idx]))
        ground_truth_list = list()
        for idx in range(len(ground_truth_dict['labels'])):
            ground_truth_list.append(_ImageDetection(score=1,
                                                     label=ground_truth_dict['labels'][idx],
                                                     boxes=ground_truth_dict['boxes'][idx]))

        count_true_positives[class_name] = 0

        tp = [0] * len(detections_with_certain_class)
        fp = [0] * len(detections_with_certain_class)

        for i, elem in tqdm(enumerate(detections_with_certain_class)):
            ovmax = -1
            gt_match = -1

            bb = elem.boxes
            for j, elem in enumerate(ground_truth_list):
                if ground_truth_list[j].label == class_name:
                    bbgt = ground_truth_list[j].boxes
                    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                                          + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = elem

            if ovmax >= iou_threshold:
                if not gt_match.used:
                    # true positive
                    tp[i] = 1
                    gt_match.used = True
                    count_true_positives[class_name] += 1
                    # update the ".json" file
                else:
                    # false positive (multiple detection)
                    fp[i] = 1
            else:
                # false positive
                fp[i] = 1

        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val

        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val

        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / ground_truth_counter_per_class[class_name]

        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

        average_precision, mean_recall, mean_precision = _voc_ap(rec[:], prec[:])
        sum_average_precision += average_precision

    mean_average_precision = sum_average_precision / len(occurring_gt_classes)

    rec_list = rec
    prec_list = prec

    return mean_average_precision, rec_list, prec_list


def calculate_map_range(ground_truth_dict: dict, result_dict: dict, iou_begin: float, iou_end: float, iou_step: float):
    """
    Gives mAP@[iou_begin:iou_end:iou_step], including iou_begin and iou_end.

    :param ground_truth_dict: dict with {boxes:, labels:}
    e.g.
    {
    'boxes':
        [[60., 80., 66., 92.],
         [59., 94., 68., 97.],
         [70., 87., 81., 94.],
         [8., 34., 10., 36.]],

    'labels':
        [2, 2, 3, 4]}
    :param result_dict: dict with {boxes:, labels:, scores:}
    e.g.
    {
    'boxes':
        [[57., 87., 66., 94.],
         [58., 94., 68., 95.],
         [70., 88., 81., 93.],
         [10., 37., 17., 40.]],

    'labels':
        [2, 3, 3, 4],

    'scores':
        [0.99056727, 0.98965424, 0.93990153, 0.9157755]}
    :param iou_begin: first iou to evaluate
    :param iou_end: last iou to evaluate (included!)
    :param iou_step: step size
    :param allow_cut_off: If true, there will be no exception if the number of predictions is not the same than
    the number of ground truth values. Will cut off predictions with the least scores.
    :return: mean average precision
    """

    _check_dicts_for_content_and_size(ground_truth_dict, result_dict)

    iou_list = np.arange(iou_begin, iou_end + iou_step, iou_step)

    mean_average_precision_sum = 0.

    rec_list_list = np.zeros((len(iou_list),result_dict['boxes'].shape[0]))
    prec_list_list = np.zeros((len(iou_list),result_dict['boxes'].shape[0]))

    for i, iou in enumerate(iou_list):
        print('Iou threshold: {:.2f}'.format(iou))
        mean_average_precision, rec_list, prec_list = calculate_map(ground_truth_dict, result_dict, iou)
        mean_average_precision_sum += mean_average_precision
        #rec_list_list.append(rec_list)
        #prec_list_list.append(prec_list)
        rec_list_list[i] = rec_list
        prec_list_list[i] = prec_list
        print()

    #rec_lists_arr = np.array(rec_list_list)
    #prec_lists_arr = np.array(prec_list_list)

    avg_rec_list = list(np.mean(rec_list_list, axis=0))
    avg_prec_list = list(np.mean(prec_list_list, axis=0))

    return mean_average_precision_sum / len(iou_list), avg_rec_list, avg_prec_list


def evaluate_predicted_boxes(gt_boxes_txt_path, pred_boxes_txt_path, iou_eval_thres=0.5, 
                             plot_prec_rec=False, plot_f1=False, verbose=False):
    """Evaluate the given predicted bounding boxes w.r.t. the given ground truth ones.

    Four metrics of interest are returned: MAP score, precision, recall and f1-score.

    The ground truth boxes are stored in a `.txt` file. Each line represents a bounding box, with 
    structure 
                   `'{x_center} {y_center} {w} {h}\n'`

    The predicted truth boxes are stored in a `.txt` file. Each line represents a bounding box, with 
    structure 
                   `'{x_center} {y_center} {w} {h} {confidence}\n'`

    Parameters
    ----------
    gt_boxes_txt_path : str
        Path to the ground truth boxes `.txt` file
    pred_boxes_txt_path : str
        Path to the predicted boxes `.txt` file
    iou_eval_thres : float, optional
        Iou threshold used for matching predictions-ground truths, by default 0.4
    plot_prec_rec : bool, optional
        Whether to plot the precision-recall curve, by default False
    plot_f1 : bool, optional
        Whether to plot the f1-score curve, by default False
    verbose : bool, optional

    Returns
    -------
    map_score : float
    f1 : float
    prec : float
    rec : float
    """  

    with open(gt_boxes_txt_path) as file:
        gt_boxes = np.array([list(map(float, line.rstrip().split())) for line in file])
    ground_truth_dict = {
        'boxes': xywh2xyxy(gt_boxes[:, :4]),
        'labels': list(np.ones(gt_boxes.shape[0]))
    }

    with open(pred_boxes_txt_path) as file:
        pred_boxes = np.array([list(map(float, line.rstrip().split())) for line in file])
    result_dict = {
        'boxes': xywh2xyxy(pred_boxes[:, :4]),
        'labels': list(np.ones(pred_boxes.shape[0])),
        'scores': list(pred_boxes[:, 4])
    }

    if not isinstance(iou_eval_thres, list):
        map_score, rec_list, prec_list = calculate_map(ground_truth_dict, result_dict, iou_threshold=iou_eval_thres)
    else:
        iou_begin, iou_end, iou_step = iou_eval_thres
        map_score, rec_list, prec_list = calculate_map_range(ground_truth_dict, result_dict, iou_begin=iou_begin, iou_end=iou_end,
                                                            iou_step=iou_step)
    f1_list = [2*prec*rec/(prec+rec) for (prec, rec) in zip(prec_list, rec_list)]
    best_f1_index = np.argmax(f1_list)
    if verbose:
        print("mAP: {:.2f}".format(map_score))
        print('Last precision: {:.2f}'.format(prec_list[-1]))
        print('Last recall: {:.2f}'.format(rec_list[-1]))
        print('Last f1: {:.2f}'.format(f1_list[-1]))
        print('Best f1: {:.2f}'.format(f1_list[best_f1_index]) + f'; index: {best_f1_index}')
    #print(len(rec_list))
    #print(len(prec_list))
    f1, prec, rec = f1_list[-1], prec_list[-1], rec_list[-1]

    if plot_prec_rec:
        plt.plot(rec_list, prec_list)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid()
        plt.show()

    if plot_f1:
        plt.figure()
        plt.plot(f1_list)
        plt.axvline(x=best_f1_index, label='best_f1', color='red')
        plt.xlabel('Predicted boxes')
        plt.ylabel('F1 score')
        plt.legend()
        plt.grid()
        plt.show()

    #print(np.argmax(f1_list))

    # TODO: delete. It is a confusion matrix w.r.t. the classes, not with respect the boxes ...
    """confusion_matrix = ConfusionMatrix(nc=1, conf_thres=0.0, iou_eval_thres=iou_eval_thres)
    detections = xywh2xyxy(np.concatenate([pred_boxes, np.ones((pred_boxes.shape[0],1))], axis=1))
    labels = np.concatenate([np.ones((gt_boxes.shape[0],1)), xywh2xyxy(gt_boxes)], axis=1)
    confusion_matrix.process_batch(torch.Tensor(detections), torch.Tensor(labels))
    print(confusion_matrix.matrix)
    confusion_matrix.plot()"""

    return map_score, f1, prec, rec



############################ HYPERPARAMETERS TUNING
def plot_tuning_results(hyperparameters_list, preferred_metric_list):
  """Plot the results from the inference parameters tuning  """
  # Extract hyperparameters and metric values
  x_values = [hyperparameters[0] for hyperparameters in hyperparameters_list]
  y_values = [hyperparameters[1] for hyperparameters in hyperparameters_list]
  m_values = [m for m in preferred_metric_list]

  # Define the resolution of your grid
  resolution = 2  # Adjust this value as needed

  num_x_bins = len(set(x_values))
  num_y_bins = len(set(y_values))

  # Define the range of your hyperparameters
  x_range = (min(x_values), max(x_values))
  y_range = (min(y_values), max(y_values))

  # Create a 2D histogram
  hist, xedges, yedges = np.histogram2d(x_values, y_values, bins=[num_x_bins, num_y_bins], range=[x_range, y_range], weights=m_values)
  #print(xedges)
  plt.figure()
  # Create a heatmap
  plt.imshow(hist.T, extent=(x_range[0], x_range[1], y_range[0], y_range[1]), origin='lower', aspect='auto', cmap='viridis')

  # Add colorbar
  plt.colorbar()

  # Add labels and title
  plt.xlabel('Confidence')
  plt.ylabel('IOU nms')
  plt.title('Inference parameters tuning')

  plt.xticks(x_values)
  plt.yticks(y_values)

  # Add the m values inside the boxes
  for i in range(num_x_bins):
      for j in range(num_y_bins):
          plt.text((xedges[i] + xedges[i+1]) / 2, (yedges[j] + yedges[j+1]) / 2, f'{hist[i, j]:.3f}', color='black',
                  ha='center', va='center',fontsize=10)

  # Show the plot
  plt.savefig(fname='hyp_tun.png')   # TODO change
  plt.show()


def inference_parameters_tuning(path_to_weigths, image_path, gt_boxes_txt_path, conf_thres_list, 
                           iou_nms_thres_list, infrared=True, crop_size=CROP_SIZE, step=STEP, 
                           preferred_metric='f1', iou_eval_thres=0.4, plot=True):
    """Perform the tuning of the inference parameters.

    Namely, the inference parameters are the confidence threshold `conf_thres` and the iou NMS threshold 
    `iou_nms_thres`.

    This tuning is perform in an exaustive way. Each possible couple `conf_thres, iou_nms_thres` is tested.
    For each couple, the inference is run on the given image and the predicted boxes are computed. Then,
    the four metrics map, precision, recall, f1-score are computed. One of these metrics is the preferred
    one, and the best couple is found by maximizing this metric.

    Parameters
    ----------
    path_to_weigths : str
        Path to the weigths for the yolov7 model
    image_path : str
        Path to the image on which performing the evaluations
    gt_boxes_txt_path : str
        Path to the ground truth `.txt` boxes
    conf_thres_list : list of float
        List of confidence thresholds to test
    iou_nms_thres_list : list of float
        List of iou NMS thresholds to test
    infrared : bool, optional
        Whether to use also the infrared or only the RGB channels, by default True
    crop_size : int
        Height and width of the crops, i.e. C
    step : int
        Step used for generating the crops, i.e. the stride
    preferred_metric : str, optional
        Preferred metric to maximize, by default 'f1'
    iou_eval_thres : float, optional
        Iou threshold used for matching predictions-ground truths, by default 0.4
    plot : bool, optional
        Whether to plot the result of the tuning or not, by default True

    Returns
    -------
    hyperparameters_list : list 
        List of all the possible combinations `conf_thres, iou_nms_thres` which are tested.
    hyperparameters_metrics_dict : dict
        Dict containing all the results of the tuning.
        Four keys: 'map', 'f1', 'prec', 'rec'. Each key contains the list of the values that metric fo all
        the possible combinations of parameters, namely `hyperparameters_list`
    """

    hyperparameters_list = [(conf_thres, iou_nms_thres) for conf_thres in conf_thres_list for iou_nms_thres in iou_nms_thres_list]
    print(hyperparameters_list)
    hyperparameters_metrics_dict = {
        'map': [],
        'f1': [],
        'prec': [],
        'rec': []
    }
    for conf_thres, iou_nms_thres in hyperparameters_list:
        print(f'START HYPERPARAMETERS conf_thres:{conf_thres} iou_nms_thres:{iou_nms_thres}')
        predict_on_img(path_to_weigths, image_path, output_bbs_path='./boxes.txt', infrared=infrared, 
                    crop_size=crop_size, step=step, conf_thres=conf_thres, iou_nms_thres=iou_nms_thres, 
                    intersect_thres=0.75, verbose=True, save_fig_pred_bbs=False)
        map_score, last_f1, last_prec, last_rec = evaluate_predicted_boxes(gt_boxes_txt_path, 
                                                                        pred_boxes_txt_path='./boxes.txt',
                                                                        iou_eval_thres=iou_eval_thres, 
                                                                        plot_prec_rec=False, 
                                                                        plot_f1=False,
                                                                        verbose=True)
        print(f'mAP:{map_score} f1:{last_f1} prec:{last_prec} rec:{last_rec}')
        hyperparameters_metrics_dict['map'].append(map_score)
        hyperparameters_metrics_dict['f1'].append(last_f1)
        hyperparameters_metrics_dict['prec'].append(last_prec)
        hyperparameters_metrics_dict['rec'].append(last_rec)
        print(f'END HYPERPARAMETERS conf_thres:{conf_thres} iou_nms_thres:{iou_nms_thres}')
        print()

    preferred_metric_list = hyperparameters_metrics_dict[preferred_metric]
    best_hyperparameters_index = np.argmax(preferred_metric_list)
    best_conf, best_iou_nms = hyperparameters_list[best_hyperparameters_index]
    best_map = hyperparameters_metrics_dict['map'][best_hyperparameters_index]
    best_f1 = hyperparameters_metrics_dict['f1'][best_hyperparameters_index]
    best_prec = hyperparameters_metrics_dict['prec'][best_hyperparameters_index]
    best_rec = hyperparameters_metrics_dict['rec'][best_hyperparameters_index]
    print(f'BEST HYPERPARAMETERS conf_thres:{best_conf} iou_nms_thres:{best_iou_nms}')
    print(f'mAP:{best_map} f1:{best_f1} prec:{best_prec} rec:{best_rec}')

    if plot:
        plot_tuning_results(hyperparameters_list, preferred_metric_list)

    return hyperparameters_list, hyperparameters_metrics_dict
