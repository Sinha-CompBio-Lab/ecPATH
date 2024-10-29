import concurrent.futures
import glob
import multiprocessing
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
import torch
from openslide import OpenSlide
from param import *
from src.gene_expr_pred import GeneExpressionPredictor
from src.preprocess import preprocess
from src.utils.utils import get_max_workers
from src.utils.utils_color_norm import *
from src.utils.utils_preprocessing import *

#
init_random_seed()
max_workers = get_max_workers()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# check for the input files, in the input directory with the input keyword
input_files = glob.glob(
    os.path.join(
        basic_param["input_dir"],
        f"*{preprocess_param['input_keyword']}*{preprocess_param['slide_extention']}",
    )
)

print(
    f'\n A total of {len(input_files)} input files found at {basic_param["input_dir"]} \n'
)

###
### preprocessing: patching (+plot), normalization/filtering, feature extraction
print("Start Preprocessing...\n")
preprocessing = preprocess(
    BASE_DIR,
    input_files,
    preprocess_param,
    feature_extraction_param,
    device,
    max_workers,
)
preprocessing.set_up()
preprocessing.process_tiles()  # process the tiles
print("Done.\n")

###
### gene expression prediction

print("Start Gene Expression Prediction...")
gene_expresion_output_path = os.path.join(
    basic_param["output_dir"],
    f"gene_expression_predictions_{feature_extraction_param['pretrained_model_name']}.csv",
)
gene_expr_predictor = GeneExpressionPredictor(
    input_files,
    model_path,
    feature_extraction_param["pretrained_model_name"],
    basic_param["cancer_type"],
    preprocess_param["slide_extention"],
    device,
)
gene_expr_predictor.load_features()
gene_expr_predictor.predict()
predictions = gene_expr_predictor.get_predictions_df()
predictions.index = [os.path.basename(file) for file in input_files]
predictions.to_csv(
    gene_expresion_output_path,
    index=True,
    sep=",",
)
print(f"Done, predictions saved as {gene_expresion_output_path}\n")
