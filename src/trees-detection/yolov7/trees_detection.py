import shutil
import sys
import argparse
import json
import os
from os.path import join, basename, isdir, isfile, dirname, splitext
from glob import glob
import time

import numpy as np
from PIL import ImageDraw, Image

from trees_detection_utils.inference import predict_on_img 
from trees_detection_utils.georeferencing import georeference


Image.MAX_IMAGE_PIXELS = 400000000


# Supported file extensions
VALID_IMG_EXT = ['.tiff', '.tif', '.TIFF', '.TIF']


best_model_dict_infrared = {
    'best_weigths_path' : os.path.join('../weights', 'best_infrared_fulltrain.pt'),
    'best_conf_thres' : 0.3,
    'best_iou_nms_thres' : 0.6
}

best_model_dict_rgb = {
    'best_weigths_path' : os.path.join('../weights', 'best_rgb_fulltrain.pt'),
    'best_conf_thres' : 0.25,
    'best_iou_nms_thres' : 0.45
}

best_model_dict_2011 = {
    'best_weigths_path' : os.path.join('../weights', 'best_rgb_fulltrain_2011.pt'),
    'best_conf_thres' : 0.3,
    'best_iou_nms_thres' : 0.6
}



# Command line arguments
parser = argparse.ArgumentParser(description='Script for the detection of trees')
parser.add_argument('input_path', metavar='input_path', type=str,
                    help='Input image file or folder of images')
parser.add_argument('--mode', dest='mode', type=str, choices=['infrared', 'rgb', '2011'],
                    default='infrared', help='modality of inference, either \'infrared\', \'rgb\' or \'2011\'. By default \'infrared\'.')
parser.add_argument('--output-folder-path', dest='output_folder_path', type=str, default='./out',
                    help='Output folder.')
parser.add_argument('--conf-thres', dest='conf_thres', type=float, default=None,
                    help='Confidence threshold used for filtering the bounding boxes.')
parser.add_argument('--iou-nms-thres', dest='iou_nms_thres', type=float, default=None,
                    help='Iou NMS threshold used for filtering the bounding boxes.')
parser.add_argument('--save-fig', dest='save_fig', action='store_true', 
                    help='Whether to save the input figure with the predicted bounding boxes or not.')
parser.add_argument('--verbose', action='store_true', 
                    help='Whether to be fully verbose or not.')


# RUN ON CPU: ~20minutes (depends on the number of trees), ~2GB RAM  
def main():
    print('Welcome to the trees detection script!')

    # Relative path from the working dir to this called script
    relative_path_to_script = dirname(sys.argv[0])

    args = parser.parse_args()
    input_path = args.input_path
    output_folder_path = args.output_folder_path
    mode = args.mode
    conf_thres, iou_nms_thres = args.conf_thres, args.iou_nms_thres
    verbose = args.verbose
    save_fig = args.save_fig

    #output_folder_path = join(os.getcwd(), output_folder_path)
    if mode=='infrared':
        infrared = True
        best_model_dict = best_model_dict_infrared
    elif mode=='rgb':
        infrared = False
        best_model_dict = best_model_dict_rgb 
    elif mode=='2011':
        infrared = False
        best_model_dict = best_model_dict_2011
    else:
        raise ValueError(f'Not valid mode {mode}')

    # Load the pre-trained model  
    best_conf_thres = best_model_dict['best_conf_thres']
    best_iou_nms_thres = best_model_dict['best_iou_nms_thres']
    best_weigths_path = os.path.join(relative_path_to_script, best_model_dict['best_weigths_path'])
    if conf_thres is None:
        conf_thres = best_conf_thres
    if iou_nms_thres is None: 
        iou_nms_thres = best_iou_nms_thres

    # Load the image/images
    images = []
    #output_folder_path = "."
    if isdir(input_path):
        #for ext in VALID_IMG_EXT:
        #    images += glob(join(input_path, f'*{ext}'))
        #print([file_name.split('.')[-1] for file_name in os.listdir(input_path)])
        images = [os.path.join(input_path, file_name) for file_name in sorted(os.listdir(input_path)) if splitext(file_name)[1] in VALID_IMG_EXT]
        #output_folder_path = join(input_path, 'out')
        print(f'Found {len(images)} images in specified input folder. ')
        #print(images)
    elif isfile(input_path):
        file_extension = splitext(input_path)[1] #input_path.split('.')[-1] 
        #output_folder_path = join(dirname(input_path), 'out')
        if file_extension in VALID_IMG_EXT:
            images = [input_path]
        else:
            print(f'File extension {file_extension} is not supported.')
    #else:
    #    print('No valid input files found.')

    if len(images) > 0:
        os.makedirs(output_folder_path, exist_ok=True)
    else:
        print('No valid input files found.')

    # Loop on test images
    for img_full_path in images:
        #img_basename, img_path = basename(img_full_path).split(".")[0], dirname(img_full_path)
        img_basename, img_path = splitext(basename(img_full_path))[0], dirname(img_full_path)

        print('\nProcessing image:', img_basename, "...")
        #img = np.array(Image.open(img_full_path))

        output_bbs_path = os.path.join(output_folder_path, f'{img_basename}_boxes.txt')
        fig_path = os.path.join(output_folder_path, f'{img_basename}_boxes.jpeg')
        try:
            # Split the image into crops, apply the model on each crop, merge the computed bounding boxes
            # and filter them
            predict_on_img(path_to_weigths=best_weigths_path, image_path=img_full_path, 
                           infrared=infrared, output_bbs_path=output_bbs_path, conf_thres=conf_thres, 
                           iou_nms_thres=iou_nms_thres, verbose=verbose, save_fig_pred_bbs=save_fig, 
                           fig_path=fig_path) 
            print() 
            
            # Georeferencing the predicted bounding boxes.
            # This results in creating two geojson files : one containing the center points, the other 
            # containing the boxes.
            print('Georeferencing ...')
            points_geojson_output_path = os.path.join(output_folder_path, f'{img_basename}_treesPoints.geojson')
            boxes_geojson_output_path = os.path.join(output_folder_path, f'{img_basename}_treesBoxes.geojson')
            tfw_path = splitext(img_full_path)[0] + '.tfw'
            georeference(output_bbs_path, tfw_path, points_geojson_output_path, boxes_geojson_output_path)
            print(f'Saved georeferenced points files: {points_geojson_output_path}')
            print(f'Saved georeferenced boxes files: {boxes_geojson_output_path}')
            print()

            print('Done image', img_basename, ".")
            print()
        finally:
            time.sleep(1)
            if os.path.exists(output_bbs_path):
                os.remove(output_bbs_path)
        

    #os.remove('traced_model.pt')
    
    print()
    print('FINISH.')


if __name__ == '__main__':
    main()
