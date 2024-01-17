# trees-detection
Detection of trees from  ortophoto.

Script `trees_detection.py`.

## Environment
Create and activate an environment with inside `requirements.txt`.

## Usage
```sh
python yolov7\trees_detection.py "dataset\19K53\19K53.tif"
```

More complete usage.
```sh
python yolov7\trees_detection.py "dataset\19K53\19K53.tif" --mode rgb --output-folder-path ./outputs --save-fig
```

Example with 2011.
```sh
python yolov7\trees_detection.py "dataset_2011\19K53\19K53.tif" --mode 2011 --output-folder-path ./outputs --save-fig
```

With a folder of images.
```sh
python yolov7\trees_detection.py "images_folder" --mode rgb --output-folder-path ./out_folder --save-fig
```

## Output
In the output folder, the following files are saved. Both of them are georeferenced `.geojson` files, meaning that they are expressed in geographic coordinates (and not pixel coordinates).
- **Points geojson file**. File containing the georeferenced baricenter (i.e. the center) of each bounding box. Example of file name : `19K53_treesPoints.geojson`.  
- **Boxes geojson file**. File containing the georeferenced boxes. It contains also the confidence scores among the properties. Example of file name : `19K53_treesBoxes.geojson`.  

Optionally, also the **bounding boxes figure** can be created. This is a `.jpeg` image, showing all trees bounding boxes. Example of file name : `19K53_boxes.jpeg`.

## General description
1. The input image is splitted into crops. 
2. On each crop, the pre-trained model is applied, computing the bounding boxes on each crop. Each bounding box has an associated confited score.
3. All these crops bounding boxes are unified. Then, they are filtered by using a fixed confidence threshold. Then, the NMS is applied, with a fixed iou NMS threshold.
4. Finally, the bounding boxes are georeferenced, generating the **points geojson file** and the **boxes geojson file**.

## Arguments 
- `input_path`: path to the input image or path to the folder containing the input images. Supported image extensions: `.tiff`, `.tif`, `.TIFF`, `.TIF`. Obviously, also the corresponding `.tfw` must be provided in the same folder.
- `--mode` : modality of inference, either `'infrared'`, `'rgb'` or `'2011'`. By default `'infrared'`. 
- `--output-folder-path`: folder into which putting the results. By default `./out`.
- `--conf-thres`: confidence threshold used for filtering the bounding boxes. By default, the best value found with the inference parameters tuning is used (this value depends on `mode`).
- `--iou-nms-thres`: iou nms threshold used for filtering the bounding boxes. By default, the best value found with the inference parameters tuning is used (this value depends on `mode`).
- `--save-fig` : whether to save the input figure with the predicted bounding boxes or not.
- `--verbose` : whether to be fully verbose or not. If not fully verbose, only a single progress bar is shown. If fully verbose, each step of the process is shown indipendently with a progress bar.

## Implementation details

The model is YoloV7, taken from this [repo](https://github.com/WongKinYiu/yolov7).

So, the starting point of our code is the `yolov7` folder. Then, it has been modified and enlarged.
- The files `train.py`, `detect.py`, `utils\datasets.py`, `utils\general.py`, `utils\metrics.py` and `utils\torch_utils.py` have been modified. The main reason of this modification is to make the code and the network compliant with the usage of the additional infrared fourth channel.
- The files `data.yaml`, `data_fulltrain.yaml` and `cfg\training\yolov7-tree.yaml` have been modified/added, into the code. This is for specifying the configurations for our specific trees detection training. (Basically, they just specify that there is just one output class. The definition of the network is the same.).
- The files inside the folder `trees_detection_utils` and the script `trees_detection.py` have been added into the code. They implement the specific operations of our trees detection project.
- The file `yolov7_training.pt` represents the starting yolov7 pre-trained weigths. Basically, we fine-tune on these. 

## Additional detail : generation of temporary files 
While the inference process is running, several temporary files are generated and saved. This is due to how yolov7 is implemented. In the end, all these temporary files are deleted.

## Repository structure

    .
    ├── dataset                   # Training dataset folder (2017)
    ├── dataset_2011              # Training dataset folder (2011)        
    ├── yolov7                    # All the code is inside this folder
    │   ├── cfg\training               # Folder containing the configurationsof the yolov7 models   
    │   ├── utils                      # Folder containing the general yolov7 utils
    │   ├── trees_detection_utils      # Folder containing the specific trees detection utils
    │   ├── data.yaml                  # Training configs (3 train images, 1 val)
    │   ├── data_fulltrain.yaml        # Full training configs (4 train images, 1 test)
    │   ├── trees_detection.py         # Trees detection script
    │   ├── train.py                   # yolov7 training script
    │   └── yolov7_training.pt         # Starting yolov7 pre-trained weigths (we fine tune on these)
    ├── weights                                 # Weigths of the best models
    ├── Trees_detection INFRARED.ipynb          # Code used for training and evaluating the infrared trees detection models
    ├── Trees_detection RGB.ipynb               # Code used for training and evaluating the rgb trees detection models
    └── README.md

