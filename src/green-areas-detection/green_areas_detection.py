import sys
import argparse
import json
import os
from os.path import join, basename, isdir, isfile, dirname, splitext
from glob import glob

import numpy as np
from PIL import ImageDraw, Image

from models.green_areas_detection_models import build_attunet

from utils.utils import predict_on_img, get_palette, get_polylines
from utils.utils import CROP_SIZE
from utils.georeferencing import georeference



Image.MAX_IMAGE_PIXELS = 400000000


# Supported file extensions
VALID_IMG_EXT = ['.tiff', '.tif', '.TIFF', '.TIF']


best_model_dict_infrared = {
    'build_model_function' : build_attunet,
    'best_weigths_path' : os.path.join('weights', 'ATT_UNET.29-0.9597 INFRARED.hdf5'),
    #'best_threshold' : 0.5 #0.45
}

best_model_dict_rgb = {
    'build_model_function' : build_attunet,
    'best_weigths_path' : os.path.join('weights', 'ATT_UNET.20-0.9523 RGB.hdf5'),
    #'best_threshold' : 0.5 #0.45
}

best_model_dict_2011 = {
    'build_model_function' : build_attunet,
    'best_weigths_path' : os.path.join('weights', 'ATTUNET_best_rgb_2011_0.9425.hdf5'),
    #'best_threshold' : 0.5 #0.45
}



# Command line arguments
parser = argparse.ArgumentParser(description='Script for the detection of green areas')
parser.add_argument('input_path', metavar='input_path', type=str,
                    help='Input image file or folder of images')
parser.add_argument('--mode', dest='mode', type=str, choices=['infrared', 'rgb', '2011'],
                    default='infrared', help='Modality of inference, either \'infrared\', \'rgb\' or \'2011\'. By default \'infrared\'.')
parser.add_argument('--output-folder-path', dest='output_folder_path', type=str, default='./out',
                    help='Output folder.')
parser.add_argument('--batch-size', dest='batch_size', type=int, default=4,
                    help='Batch size used for paralellizing the computation.')
parser.add_argument('--threshold', dest='threshold', type=float, default=0.5,
                    help='Threshold used for classifying the model output scores.')
parser.add_argument('--tolerance', dest='tolerance', type=float, default=1.0,
                    help='Tolerance used for computing the polylines')
parser.add_argument('--min-area', dest='min_area', type=int, default=10,
                    help='Minimum area used for computing the polylines')
parser.add_argument('--save-mask', action='store_true', 
                    help='Whether to save the binary mask or not.')


# RUN ON CPU: ~20minutes, ~3GB RAM
def main():
    print('Welcome to the green areas detection script!')

    # Relative path from the working dir to this called script
    relative_path_to_script = dirname(sys.argv[0])

    args = parser.parse_args()
    input_path = args.input_path
    output_folder_path = args.output_folder_path
    batch_size, threshold = args.batch_size, args.threshold
    tolerance, min_area = args.tolerance, args.min_area
    mode = args.mode
    save_mask = args.save_mask

    #output_folder_path = join(os.getcwd(), output_folder_path)
    if mode=='infrared':
        n_input_channels = 4
        best_model_dict = best_model_dict_infrared
    elif mode=='rgb':
        n_input_channels = 3
        best_model_dict = best_model_dict_rgb 
    elif mode=='2011':
        n_input_channels = 3
        best_model_dict = best_model_dict_2011
    else:
        raise ValueError(f'Not valid mode {mode}')

    # Load the pre-trained model  
    build_model_function = best_model_dict['build_model_function']
    #best_threshold = best_model_dict['best_threshold']
    best_weigths_path = os.path.join(relative_path_to_script, best_model_dict['best_weigths_path'])
    model = build_model_function(input_shape=(CROP_SIZE,CROP_SIZE,n_input_channels))
    model.load_weights(best_weigths_path)
    #if threshold is None:
    #    threshold = best_threshold

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
        #_, file_extension = splitext(input_path)
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

    # Loop on the images
    for img_full_path in images:
        #img_basename, img_path = basename(img_full_path).split(".")[0], dirname(img_full_path)
        img_basename, img_path = splitext(basename(img_full_path))[0], dirname(img_full_path)

        print('\nProcessing image:', img_basename, "...")
        img = np.array(Image.open(img_full_path))[:, :, :n_input_channels]

        # Split the image into crops, apply the model on each crop, merge the crops predictions
        # into a single big output predictions image, containing scores in [0,1]
        preds = predict_on_img(model=model, img=img, batch_size=batch_size)
        print()

        # Thresholded binary predictions: either 0 or 1. 
        # This is our binary mask.
        binary_mask = (preds>threshold).astype('uint8')
        if save_mask:
            # Save the binary mask as png image
            binary_mask_png = Image.fromarray(binary_mask[:, :, 0])
            binary_mask_png.putpalette(get_palette())  # 0 remains 0, 1 becomes 255
            binary_mask_png.save(join(output_folder_path, f"{img_basename}_binary_mask.png"))
            print(f'Binary mask saved in {join(output_folder_path, f"{img_basename}_binary_mask.png")}')

        print('Vectorizing and georeferencing ...')

        # Compute the poly-lines (i.e. vectorization)
        poly_lines = get_polylines(binary_mask, tolerance=tolerance, min_area=min_area)
        #polylines_json_path = join(output_folder_path, f"{img_basename}_polylines.json")
        # Save the poly-lines as json 
        #with open(polylines_json_path, "w") as f:
        #    json.dump({'objects': poly_lines}, f)
        #print(f'Poly-lines file saved in {join(output_folder_path, f"{img_basename}_polylines.json")}')

        # Georeferencing the poly-lines, creating the `.geojson` file.
        tfw_path = splitext(img_full_path)[0] + '.tfw'
        polylines_geojson_path = join(output_folder_path, f"{img_basename}_polylines.geojson")
        georeference(poly_lines, tfw_path, polylines_geojson_path)
        print(f'Georeferenced poly-lines file saved in {polylines_geojson_path}')

        # Compute the polylines mask
        """polyline_mask = np.zeros(shape=binary_mask.shape[:2], dtype=np.uint8)
        polyline_mask = Image.fromarray(polyline_mask)
        draw_polyline = ImageDraw.Draw(polyline_mask)
        for p in poly_lines:
            draw_polyline.polygon(p, fill=1)
        # Save the polylines mask as png image
        polyline_mask.putpalette(get_palette()) # 0 remains 0, 1 becomes 255
        polyline_mask.save(join(output_folder_path, f"{img_basename}_polylines.png"))
        polyline_mask = np.array(polyline_mask)
        print(f'Poly-lines figure saved in {join(output_folder_path, f"{img_basename}_polylines.png")}')"""

        print('Done image', img_basename, ".")
        print()


    print()
    print('FINISH.')


if __name__ == '__main__':
    main()
