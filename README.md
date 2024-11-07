
# ecPATH: Predicting ecDNA status in Tumors from Histopathology Slide Images

### Please note:
1. In this initial release, you need to manually download ecPATH model weights from google drive 'https://drive.google.com/file/d/16_YdmAcALNeITmUMEXBQGYCTLu2UYccM/view?usp=sharing', 
decompress it and put entire ***Data*** folder on the top level of this directory.
2. Our pipeline uses resnet50 model as the default feature extraction agent, to use UNI model, you need to obtain approval (an access token) from [Hugging Face](https://huggingface.co/MahmoodLab/UNI) and copy the access token in `./Prediction/param.py`

## Usage & Demo:  
#### 0. Install conda (if needed): [install conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
#### 1. Clone this repo to you local
    git clone https://github.com/Sinha-CompBio-Lab/ecPATH.git
#### 2. Create the desired conda environment
    conda env create -f environment.yml
#### 3. Customize `./Prediction/param.py` to fit your analysis.(very important)
    key parameters are critical. Minimum: provide cancer_type, slide_extention, input_keyword, pretrained_model_name.
#### 4. Prepare input slides: place slides in `./Prediction/input/` (Currently, only `.svs` image files are tested & supported)
    make sure to include a keyword in a set of input slides, e.g., test_1.svs, test_2.svs, ..., test_n.svs.
#### 5. Execute prediction script
    python3 ./Prediction/predict.py
### 6. Output:
        a.) for each input slide:
            1.) a collection of tile features at `./Prediction/input/SlideKeyword_1/_features/features_{model_name}.npy` 
            2.) a visualization of tile selection at `./Prediction/input/SlideKeyword_2/_masks/mask.pdf`
            3.) a tile coordinates list at `./Prediction/input/SlideKeyword_3/_coordinates/tile_coordinates.csv`
        b.) for each analysis:
            1.) intermediate gene expresion predictions (for each input slide) at ./Prediction/output/{cancer_type}_gene_expression_predictions_{model_name}.csv
            2.) final ecDNA prediction (for each input slide) at ./Prediction/output/{cancer_type}_ecDNA_predictions_{model_name}.csv
        
## Development
1. This pipeline is tested on 'Ubuntu 22.04.3 LTS' with GPU available (NVIDIA GeForce RTX 3090).
2. Conda is needed for environment management. Current pipeline is developed & tested under conda 24.5.0.

### Development log
#### 11/07/ 2024 update plotting scripts & data
- Function description:<br />
    i. Mudra updated scripts & data for reproduce the figures in the manuscripts

#### 11/06/ 2024 added ecDNA prediction threshold, added google drive data container 
- Function description:<br />
    i. added threshold for calling ecDNA positive/negative using Youden J statistic threshold
    ii.added capability of downloding model weights from google drive for testing/reviewing (before Zenodo)

#### 10/29/ 2024 added gene expression & ecDNA prediction steps: 
- Function description:<br />
    1. Performs downstream gene expression prediction and ecDNA prediction.
    2. Will output `./Prediction/output/{cancer_type}_ecDNA_predictions_{model_name}.csv` and `./Prediction/output/{cancer_type}_gene_expression_predictions_{model_name}.csv`

#### 10/27/ 2024 added preprocessing steps: 
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