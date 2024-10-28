
# ecPATH: Predicting ecDNA status in Tumors from Histopathology Slide Images

## Table of Content

## TBD

## Usage & Demo:
    1. clone this repo.  
    2. create an conda environment `conda env create -f environment.yml`
    3. Prepare input slides: place slides in `./Prediction/input/` and include a common keyword in the names. Currently, only SVS image files (`.svs` extension) are supported.
    4. execute `python3 ./Prediction/predict.py` (as of 10/27/2024, only preprocessing is implemented.)

## development log

###  10/27/ 2024 added preprocessing steps: 

- Function description:<br />
    1. Given any input slides placed in `./Prediction/input/`, the program will perform preprocessing.  
        1. creating tiles given the tile size specified in `./Prediction/param.py`.  
        2. tile normalization and evalution, keep only informative tiles. 
        3. generate a pdf for WSI & tiles visualization.  
        4. Feature extraction at the tile level using pretrained models.  
            1.) ResNet50 (potentially generalize to any models from torchvision).  
            2.) UNI (potentially generalize to any models from huggingface).  
    2. Each input slide should be placed in `./Prediction/input/` and have a **keyword** and an **extention** (only svs is supported in this version) in the names, e.g., `./AnyKeyword_1.svs`, `AnyKeyword_2.svs`, ..., `AnyKeyword_n.svs`, keyword will be used for include all slides in one analysis.
    3. For any of the input slide, after preprocessing, there will be `./Prediction/input/AnyKeyword_1/_coordinates/tile_coordinates.csv`, `./Prediction/input/AnyKeyword_1/_masks/mask.pdf` and `./Prediction/input/AnyKeyword_1/_features/features_{model_name}.npy` generated for each slide.
    4. Users may use any of the two models supported in this version: resnet50 and UNI.
        1. Please note that to use UNI model, you need to make a request [here](https://huggingface.co/MahmoodLab/UNI) and provide an access token in `./Prediction/param.py`.

