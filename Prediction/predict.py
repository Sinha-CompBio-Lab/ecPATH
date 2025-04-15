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

print("###########################################################################\n")
print("# ecPATH is a tool for predicting ecDNA from H&E-stained pathology slides #\n")
print("###########################################################################\n")

print("Setting up the environment...\n")
import numpy as np
import openslide
import pandas as pd
import torch
from openslide import OpenSlide
from param import Config
from src.ec_dna_pred import ecDNA_Predictor
from src.gene_expr_pred import GeneExpressionPredictor
from src.preprocess import preprocess
from src.utils.utils import get_max_workers
from src.utils.utils_preprocessing import init_random_seed
import argparse
import pyreadr


def main():
    if not os.path.exists(MLP_model_path) or not os.path.exists(LR_model_path):
        print("Model weights not found, start downloading now...")
        if config.data_cloud_param["zenodo_record_id"] != "":
            print("Downloading from Zenodo...")
            subprocess.run(["zenodo_get", config.data_cloud_param["zenodo_record_id"]], check=True)
            print("Done.\n")

            # Unzip the downloaded file
            zip_path = os.path.join(config.BASE_DIR, "..", "Data.zip")
            if os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(".")
                os.remove(zip_path)
            else:
                warnings.warn(f"{zip_path} not found after download")
        else:
            print("Downloading from Google Drive... (NOT implemented yet)")
            print( "Please use zenodo instead")
            # gdown.download(data_cloud_param["gcloud_drive"], DATA_DIR, fuzzy=True)

    # create directory for input and output if not exist
    os.makedirs(config.basic_param["input_dir"], exist_ok=True)
    os.makedirs(config.basic_param["output_dir"], exist_ok=True)
    print("Setting up environment Done!\n")


    ###
    ### preprocessing: patching (+plot), normalization/filtering, feature extraction
    if not config.preprocces_flag:
        feature_extract_dict = {"uni":"-uni.npy", "resnet":".npy"}
        feature_extinson = feature_extract_dict[args.feature_extract]
        #input_files = glob.glob(f'{config.basic_param["input_dir"]}/*{feature_extinson}', recursive=True) # None Lihe
        input_files = glob.glob(f'{args.features_stored}/{args.cancer_type}/*/_features/*{feature_extinson}', recursive=True) # Lihe
    
        print("preprocces_flag set False, skiping Preprocessing...\n")
        print("Will use features in input directory\n")
    else:
        # check for the input files, in the input directory with the input keyword
        input_files = glob.glob(
            os.path.join(
                config.basic_param["input_dir"],
                f"*{config.preprocess_param['input_keyword']}*{config.preprocess_param['slide_extention']}",
            )
        )
        print("preprocess_flag set True. Start Preprocessing...")
        preprocessing = preprocess(
            config.BASE_DIR,
            input_files,
            config.preprocess_param,
            config.feature_extraction_param,
            device,
            max_workers,
        )
        preprocessing.set_up()
        preprocessing.process_tiles()  # process the tiles
        print("Done.\n")

    print(
        f'A total of {len(input_files)} input files found at {config.basic_param["input_dir"]} \n'
    )
    breakpoint()
    return input_files


def gene_prediction(input_files):
    ###
    ### gene expression prediction
    print("Start Gene Expression Prediction...")
    gene_expresion_output_path = os.path.join(
        config.basic_param["output_dir"],
        f"{config.basic_param['cancer_type']}_gene_expression_predictions_{config.feature_extraction_param['pretrained_model_name']}.csv",
    )
    gene_expr_predictor = GeneExpressionPredictor(
        input_files,
        MLP_model_path,
        config.feature_extraction_param["pretrained_model_name"],
        config.basic_param["cancer_type"],
        config.preprocess_param["slide_extention"],
        device,
    )
    gene_expr_predictor.load_features(config.preprocces_flag)
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

    return gene_expr_predictions

def ecDNA_prediction(gene_expr_predictions):
    ##
    ## ecDNA prediction
    print("Start ecDNA Prediction...")
    ecDNA_output_path = os.path.join(
        config.basic_param["output_dir"],
        f"{config.basic_param['cancer_type']}_ecDNA_predictions_{config.feature_extraction_param['pretrained_model_name']}.csv",
    )
    ecDNA_Prediction = ecDNA_Predictor(
        gene_expr_predictions,
        LR_model_path,
        config.feature_extraction_param["pretrained_model_name"],
        config.basic_param["cancer_type"],
    )
    ecDNA_predictions = ecDNA_Prediction.predict()

    ecDNA_predictions_df = pd.DataFrame(np.mean(ecDNA_predictions, axis=0))
    ecDNA_predictions_df.columns = ["ecDNA_score"]
    ecDNA_predictions_df.index = gene_expr_predictions.index
    ecDNA_predictions_df.index.name = "input_slide"

    if config.ecDNA_param.get("threshold", {}).get(config.basic_param.get("cancer_type")) is None:
        print(
            f"ecDNA prediction threshold not found for {config.basic_param['cancer_type']}, considering set the threshold in param.py"
        )
        print("ecDNA predictions are saved without binary prediction\n")
    else:
        ecDNA_predictions_df["ecDNA_prediction"] = (
            ecDNA_predictions_df["ecDNA_score"]
            > config.ecDNA_param["threshold"][config.basic_param["cancer_type"]]
        )

    ecDNA_predictions_df.to_csv(
        ecDNA_output_path,
        index=True,
        sep=",",
    )
    print(f"Done, ecDNA predictions saved as {ecDNA_output_path}\n")

    print("Thanks for using ecPATH!\n")


parser = argparse.ArgumentParser(description='train gene expression')
parser.add_argument('--cancer_type', type=str,default='STAD',choices=['BRCA','LUAD','STAD']) # Add others later
parser.add_argument('--feature_extract', type=str, default= 'uni', choices=['uni','resnet'])
parser.add_argument('--base_input_path', type=str, default="/shares/sinha/sadeleye/ecPATH",
                    help="base input path where all data is kept")
parser.add_argument('--features_stored', type=str, default='/shares/sinha/lliu/projects/pre-cancer-image-omics/rawData/slides')
parser.add_argument('--output_path', type=str, default='/shares/sinha/sadeleye/ecPATH_Results')
parser.add_argument('--preprocces_flag',type=bool, default=False)
args = parser.parse_args()


config = Config(args)
MLP_model_path = os.path.join(config.DATA_DIR, "Model", "MLP")
LR_model_path = os.path.join(config.DATA_DIR, "Model", "LR")

if __name__ == '__main__':
    init_random_seed()
    max_workers = get_max_workers()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_files = main()
    gene_expression = gene_prediction(input_files)
    ecDNA_prediction(gene_expression)