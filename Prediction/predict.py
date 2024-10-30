import concurrent.futures
import glob
import multiprocessing
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
import torch
from openslide import OpenSlide
from param import *
from src.ec_dna_pred import ecDNA_Predictor
from src.gene_expr_pred import GeneExpressionPredictor
from src.preprocess import preprocess
from src.utils.utils import get_max_workers
from src.utils.utils_color_norm import *
from src.utils.utils_preprocessing import *

#
init_random_seed()
max_workers = get_max_workers()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(BASE_DIR, "..", "Data", "Model")
MLP_model_path = os.path.join(model_path, "MLP")
LR_model_path = os.path.join(model_path, "LR")

# create directory for input and output if not exist
os.makedirs(basic_param["input_dir"], exist_ok=True)
os.makedirs(basic_param["output_dir"], exist_ok=True)

# # check for Data directory if not exist download from zenodo
# if not os.path.exists(os.path.join(BASE_DIR, "Data")):
#     print("Data directory not found, downloading from zenodo...")
#     download_data(Zenodo_record_id)

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
if MODE == "reviewer_test":
    print("In reviewer_test mode, skip Preprocessing...\n")
    print("Will use test features provided by the developper\n")
else:
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
    f"{basic_param['cancer_type']}_gene_expression_predictions_{feature_extraction_param['pretrained_model_name']}.csv",
)
gene_expr_predictor = GeneExpressionPredictor(
    input_files,
    MLP_model_path,
    feature_extraction_param["pretrained_model_name"],
    basic_param["cancer_type"],
    preprocess_param["slide_extention"],
    device,
)
gene_expr_predictor.load_features()
gene_expr_predictor.predict()
gene_expr_predictions = gene_expr_predictor.get_predictions_df()
gene_expr_predictions.index = [os.path.basename(file) for file in input_files]
gene_expr_predictions.index.name = "input_slide"
gene_expr_predictions.to_csv(
    gene_expresion_output_path,
    index=True,
    sep=",",
)
print(f"Done, gene expression predictions saved as {gene_expresion_output_path}\n")


###
### ecDNA prediction
print("Start ecDNA Prediction...")
ecDNA_output_path = os.path.join(
    basic_param["output_dir"],
    f"{basic_param['cancer_type']}_ecDNA_predictions_{feature_extraction_param['pretrained_model_name']}.csv",
)
ecDNA_Prediction = ecDNA_Predictor(
    gene_expr_predictions,
    LR_model_path,
    feature_extraction_param["pretrained_model_name"],
    basic_param["cancer_type"],
)
ecDNA_predictions = ecDNA_Prediction.predict()

ecDNA_predictions_df = pd.DataFrame(np.mean(ecDNA_predictions, axis=0))
ecDNA_predictions_df.columns = ["ecDNA_score"]
ecDNA_predictions_df.index = gene_expr_predictions.index
ecDNA_predictions_df.index.name = "input_slide"
ecDNA_predictions_df.to_csv(
    ecDNA_output_path,
    index=True,
    sep=",",
)
print(f"Done, ecDNA predictions saved as {ecDNA_output_path}\n")
