# green-areas-detection
Detection of green areas from  ortophoto.

Script `green_areas_detection.py`.

## Environment
Create and activate an environment with inside `requirements.txt`.

## Usage
```sh
python green_areas_detection.py "dataset\19K53\19K53.tif"
```

More complete usage.
```sh
python green_areas_detection.py "dataset\19K53\19K53.tif"  --mode rgb --output-folder-path ./outputs --save-mask 
```

Example with 2011.
```sh
python green_areas_detection.py "dataset_2011\19K53\19K53.tif"  --mode 2011 --output-folder-path ./outputs_2011 --save-mask 
```

With a folder of images.
```sh
python green_areas_detection.py "images_folder"  --mode rgb --output-folder-path ./out_folder --save-mask 
```


## Output
The `.geojson` file containing the **georeferenced poly-lines** is saved in the specified output folder. Example of file name : `19K53_polylines.geojson`.  

Optionally, the **binary mask** is saved in the output folder. Binary mask containing $0/255$ values: $255$ means green area pixel, $0$ means no green area. This mask is saved as `.png` image. Example of file name : `19K53_binary_mask.png`. *Remark : this binary mask is saved before the vectorization and the creation of the poly-lines. Therefore, this image mask and the `.geojson` file do not correspond precisely.*

## General description
1. The input image is splitted into crops. 
2. On each crop, the pre-trained model is applied, computing the per-pixel predicted scores. Since the crops can be overlapping, these crop binary predictions are merged into a single big predictions array.
3. The predictions are thresholded, obtaining $0/1$ values. The result is the **binary mask**.
4. The binary mask is processed, generating the **poly-lines**. This is the vectorization.
5. Finally, the poly-lines are georeferenced, obtaining the **georeferenced poly-lines**.

## Arguments 
- `input_path`: path to the input image or path to the folder containing the input images. Supported image extensions: `.tiff`, `.tif`, `.TIFF`, `.TIF`. Obviously, also the corresponding `.tfw` must be provided in the same folder.
- `--mode` : modality of inference, either `'infrared'`, `'rgb'` or `'2011'`. By default `'infrared'`. 
- `--output-folder-path`: folder into which putting the results. By default `./out`.
- `--batch-size`: number of crops in the same batch. This is used for paralellizing the computation. By default $4$. Increasing the batch size results in a faster computation, but with higher memory consumption.
- `--threshold`: threshold used on the model output scores, for computing the $0/1$ values. By default, $0.5$ is used.
- `--tolerance`: tolerance used for computing the polylines. By default $1.$.
- `--min-area`: minimum area used for computing the polylines. By default $10$.
- `--save-mask`: whether to save the binary mask or not. By default False.


## Repository structure

    .
    ├── dataset                                          # Training dataset folder (2017)
    ├── dataset_2011                                     # Training dataset folder (2011)   
    ├── models                                           # Code for creating the models
    ├── utils                                            # Utils code
    ├── weights                                          # Weigths of the best models
    ├── TRAINING INFRARED green_areas_detection.ipynb    # Code used for training and evaluating the infrared green areas detection models
    ├── TRAINING RGB green_areas_detection.ipynb         # Code used for training and evaluating the rgb green areas detection models
    ├── green_areas_detection.py                         # Green areas detection script
    ├── shp_into_binary.ipynb                            # Code for converting shp files into binary png images
    ├── .gitignore
    └── README.md

