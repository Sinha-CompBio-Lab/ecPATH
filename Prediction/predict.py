import concurrent.futures
import glob
import multiprocessing
import os
import subprocess
import warnings
import zipfile

# import gdown

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print("###########################################################################.\n")
print("# ecPATH is a tool for predicting ecDNA from H&E-stained pathology slides #\n")
print("###########################################################################\n")

print("Setting up the environment...\n")
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
from src.utils.utils_preprocessing import init_random_seed

#
init_random_seed()
max_workers = get_max_workers()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MLP_model_path = os.path.join(DATA_DIR, "Model", "MLP")
LR_model_path = os.path.join(DATA_DIR, "Model", "LR")

if not os.path.exists(MLP_model_path) or not os.path.exists(LR_model_path):
    print("Model weights not found, start downloading now...")
    if data_cloud_param["zenodo_record_id"] != "":
        print("Downloading from Zenodo...")
        subprocess.run(["zenodo_get", data_cloud_param["zenodo_record_id"]], check=True)
        print("Done.\n")

        # Unzip the downloaded file
        zip_path = os.path.join(BASE_DIR, "Data.zip")
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(".")
            os.remove(zip_path)
        else:
            warnings.warn(f"{zip_path} not found after download")
    else:
        print("Downloading from Google Drive... (NOT implemented yet)")
        # gdown.download(data_cloud_param["gcloud_drive"], DATA_DIR, fuzzy=True)


# create directory for input and output if not exist
os.makedirs(basic_param["input_dir"], exist_ok=True)
os.makedirs(basic_param["output_dir"], exist_ok=True)
print("Setting up environment Done!\n")

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
    f'A total of {len(input_files)} input files found at {basic_param["input_dir"]} \n'
)

###
### preprocessing: patching (+plot), normalization/filtering, feature extraction
if MODE == "reviewer_test":
    print("In reviewer_test mode, skip Preprocessing...\n")
    print("Will use test features provided by the developper\n")
else:
    print("Start Preprocessing...")
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

if ecDNA_param.get("threshold", {}).get(basic_param.get("cancer_type")) is None:
    print(
        f"ecDNA prediction threshold not found for {basic_param['cancer_type']}, considering set the threshold in param.py"
    )
    print("ecDNA predictions are saved without binary prediction\n")
else:
    ecDNA_predictions_df["ecDNA_prediction"] = (
        ecDNA_predictions_df["ecDNA_score"]
        > ecDNA_param["threshold"][basic_param["cancer_type"]]
    )

ecDNA_predictions_df.to_csv(
    ecDNA_output_path,
    index=True,
    sep=",",
)
print(f"Done, ecDNA predictions saved as {ecDNA_output_path}\n")

print("Thanks for using ecPATH!\n")
