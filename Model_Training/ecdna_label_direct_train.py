import warnings

import numpy as np
import pandas as pd
import torch

np.warnings = warnings
import os
import pickle

np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

import json
import sys
import time

import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data import Dataset, Subset
import argparse
import pyreadr
from ecdna_label_residual_model import training_epoch,training_epoch_with_auc_select, EcDNATileClassifier_AucSelect, RandomForestModel_Class
from utils import Feature_Dataset, create_stratified_grouped_cv_splits
from copy import deepcopy
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from collections import defaultdict


#============================
def dataPrep_MLecdna(ecDNA_df, cancer_type):
    # Sample and gene status data
    sample_ecDNA_status_df = ecDNA_df.groupby('sample').agg({
            'gene_class': lambda x: '1' if 'circular' in set(x) else '0'
            }).reset_index()
    sample_ecDNA_status_df = sample_ecDNA_status_df.rename(columns={'gene_class': 'ecDNA_status'}) # rename to ecDna_status
    ecDNA_samples = sample_ecDNA_status_df['sample'].tolist()
    
    # Reading in TCGA slide lables by tumor type and making sure id matches a gene status data we have. 
    slides_meta_guide_df = pd.read_csv(
        os.path.join(args.base_input_path, "all_slides_TCGA_filtered.tsv"), sep="\t"
    )
    slides_meta_guide_df_type = slides_meta_guide_df[
        (slides_meta_guide_df["source_abrev"] == cancer_type)
        & (slides_meta_guide_df["slide_submitter_id"].str[:-1].isin(ecDNA_samples))
    ].reset_index(drop=True)
   
    # Combine data frames
    sample_to_status = dict(zip(
        sample_ecDNA_status_df['sample'],
        sample_ecDNA_status_df['ecDNA_status']
    ))
    slides_meta_guide_df_type['sample_id'] = slides_meta_guide_df_type['slide_submitter_id'].str[:-1]  # Remove last character
    slides_meta_guide_df_type['ecDNA_status'] = slides_meta_guide_df_type['sample_id'].map(sample_to_status)

    # Make dataset and group id. Dataset is image features and targets 
    # group_id matches dataset to image to patient id. No cross contamation in training sets of same patient. 
    all_unique_patients_slides = slides_meta_guide_df_type["case_submitter_id"].unique()
    group_ids = np.arange(len(all_unique_patients_slides))

    unique_patient_slide_ids = dict(zip(
        all_unique_patients_slides,
        group_ids 
    ))
    
    file_paths = []
    targets = []
    group_ids = []
    for idx, row in slides_meta_guide_df_type.iterrows():
        feature_extract_dict = {"uni":"-uni.npy", "resnet":".npy", "titan":".h5"}
        feature_extinson = feature_extract_dict[args.feature_extract]
        feature_extinson_titan = ".h5"
        if feature_extinson == ".h5":
            temp_feature_path = os.path.join(args.feature_input_dir,
                                                row["filename"].split(".")[0] + feature_extinson_titan,)
        else:
            temp_feature_path = os.path.join(
                    args.feature_input_dir,
                    "rawData",
                    "slides",
                    args.cancer_type,
                    row["id"],
                    "_features",
                    row["filename"].replace(
                        ".svs", feature_extinson 
                    ), 
                )
        if not os.path.exists(temp_feature_path):
            continue
        file_paths.append(temp_feature_path)
        group_ids.append(unique_patient_slide_ids[row["case_submitter_id"]])
        targets.append(row["ecDNA_status"])

    print(f" Working with this many samples: {len(file_paths)}")
    print(f"{(sum(int(t) for t in targets)/len(targets))*100:.2f}% of samples are Postive")
    return Feature_Dataset(file_paths, targets, feature_extinson), group_ids


def dataPrep_DeepPT(cancer_type,ecDNA_df):
    # ecDna status from DeepPt
    project_id = f"TCGA-{cancer_type}"
    ecDNA_cancer_df = ecDNA_df[ecDNA_df['cancer_type'] == project_id]


    # keep ecDNA status and sample names and merge with expression file
    columns_to_keep = ['sample', 'patient_id', 'ecDNA_status']
    # Create a new DataFrame with only the selected columns
    ecDNA_df_selected = ecDNA_cancer_df[columns_to_keep]
    # Drop duplicate rows based on the selected columns
    ecDNA_df = ecDNA_df_selected.drop_duplicates()
    ecDNA_samples = ecDNA_df['sample'].tolist()

    # Reading in TCGA slide lables by tumor type and making sure id matches a gene status data we have. 
    slides_meta_guide_df = pd.read_csv(
        os.path.join(args.base_input_path, "all_slides_TCGA_filtered.tsv"), sep="\t"
    )
    slides_meta_guide_df_type = slides_meta_guide_df[
        (slides_meta_guide_df["source_abrev"] == cancer_type)
        & (slides_meta_guide_df["slide_submitter_id"].isin(ecDNA_samples))
    ].reset_index(drop=True)

    # Combine data frames
    sample_to_status = dict(zip(
        ecDNA_df['sample'],
        ecDNA_df['ecDNA_status']
    ))
    slides_meta_guide_df_type['ecDNA_status'] = slides_meta_guide_df_type['slide_submitter_id'].map(sample_to_status)


    all_unique_patients_slides = ecDNA_df["patient_id"].unique()
    group_ids = np.arange(len(all_unique_patients_slides))
    unique_patient_slide_ids = dict(zip(
        all_unique_patients_slides,
        group_ids 
    ))

    file_paths = []
    targets = []
    group_ids = []
    for idx, row in slides_meta_guide_df_type.iterrows():
        feature_extract_dict = {"uni":"-uni.npy", "resnet":".npy", "titan":".h5"}
        feature_extinson = feature_extract_dict[args.feature_extract]
        feature_extinson_titan = ".h5"
        if feature_extinson == ".h5":
            temp_feature_path = os.path.join(args.feature_input_dir,
                                                row["filename"].split(".")[0] + feature_extinson_titan,)
        else:
            temp_feature_path = os.path.join(
                    args.feature_input_dir,
                    "rawData",
                    "slides",
                    args.cancer_type,
                    row["id"],
                    "_features",
                    row["filename"].replace(
                        ".svs", feature_extinson 
                    ), 
                )

        if not os.path.exists(temp_feature_path):
            continue
        file_paths.append(temp_feature_path)
        group_ids.append(unique_patient_slide_ids[row["case_submitter_id"]])
        targets.append(row["ecDNA_status"])
    print(f" Working with this many samples: {len(file_paths)}")
    try:
        postive_ratio = (sum(targets)/len(targets))*100
    except Exception as e:
            print(f"Error loading files: {e}")
            print(f"Check that path is correct. Last path: {temp_feature_path}")
            exit()
    print(f"{postive_ratio:.2f}% of samples are Postive")
    return Feature_Dataset(file_paths, targets, feature_extinson), group_ids

def cpyNumb_dataPrep_DeepPT(cancer_type, cpData_df, ecDNA_df):
    
    cpyNumb_samples = cpData_df.columns.tolist()[1:]
    cp_transposed = cpData_df.set_index('Sample').T

    # ecDna status from DeepPt
    project_id = f"TCGA-{cancer_type}"
    ecDNA_cancer_df = ecDNA_df[ecDNA_df['cancer_type'] == project_id]
    # keep ecDNA status and sample names and merge with expression file
    columns_to_keep = ['sample', 'patient_id', 'ecDNA_status']
    # Create a new DataFrame with only the selected columns
    ecDNA_df_selected = ecDNA_cancer_df[columns_to_keep]
    # Drop duplicate rows based on the selected columns
    ecDNA_df = ecDNA_df_selected.drop_duplicates()


    ecDNA_df_type = ecDNA_df[
        (ecDNA_df["sample"].str[:-1].isin(cpyNumb_samples))
    ].reset_index(drop=True)
    ecDNA_df_type['sample_id'] = ecDNA_df_type["sample"].str[:-1]
    ecDNA_df_type = ecDNA_df_type.set_index('sample_id')


    merged_df_inner = cp_transposed.join(
        ecDNA_df_type,
        how='inner'
    )

    # Not needed, all cols have variance. 
    no_variance_cols = [col for col in merged_df_inner.columns 
                      if merged_df_inner[col].nunique() == 1]


    non_feature_cols = ['sample', 'patient_id', 'ecDNA_status']
    feature_cols = [col for col in merged_df_inner.columns if col not in non_feature_cols or col in no_variance_cols]
    breakpoint()
    return merged_df_inner[feature_cols]


#============================================================================================

def nested_cv_with_regularization_grouped(dataset, input_dim, group_ids, n_outer_folds=5, 
                                         batch_size=32, epochs=30, hidden_dim=256, 
                                         learning_rate=0.001, early_stopping_patience=5,
                                         primary_metric='auc_scores'):
   
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    
    # Define regularization types and parameters to try
    # Current code is only set up for one regulization param
    regularization_params = {
        'L1 only': {'l1_values': [0.005], 'l2_values': [0.0]},
    }
    # Old Reguliztion Params - Not tested with new code
    # regularization_params = {
    #     'L1 only': {'l1_values': [0.0001, 0.0005, 0.001, 0.005, 0.01], 'l2_values': [0.0]},
    #     'L2 only': {'l1_values': [0.0], 'l2_values': [0.0001, 0.0005, 0.001, 0.005, 0.01]},
    #     'Elastic Net': {'l1_values': [0.0001, 0.001, 0.01], 'l2_values': [0.0001, 0.001, 0.01]}
    # }
 
    dataset_indices = np.arange(len(dataset))
    all_labels = [dataset[i][1] for i in range(len(dataset))]  # Adjust based on your dataset structure


    # Create OUTER CV splits respecting group constraints
    outer_splits = create_stratified_grouped_cv_splits(dataset_indices, all_labels, group_ids, n_splits=n_outer_folds)
    
    metrics = {'auc_scores': [], 'f1': [], 'recall': [], 'precision': [], 'accuracy': []} 
    all_results = {
    (l1, l2): metrics 
    for l1 in regularization_params['L1 only']['l1_values']
    for l2 in regularization_params['L1 only']['l2_values']
    }
    # Outer CV loop
    for outer_fold, (train_val_idx, test_val_idx) in enumerate(outer_splits):
        print(f"\nOuter Fold {outer_fold+1}/{n_outer_folds}")
        
        for reg_type, params in regularization_params.items():
            for l1_lambda in params['l1_values']:
                for l2_lambda in params['l2_values']:
                    if l1_lambda == 0.0 and l2_lambda == 0.0:
                        continue  # Skip no regularization case
                    
                    print(f"  Testing L1={l1_lambda}, L2={l2_lambda}")
                
                    feature_select_type = {'percent': 0.1} # {'threshold': 0.7}, {'topK':10} , {'percent': 0.1}

                    # Train a model with best parameters on all train_val data (except early stopping subset)
                    if args.
                    model = EcDNATileClassifier_AucSelect(input_dim, args.feature_extract, hidden_dim, feature_select_type).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                    if args.model_feature_select == 'global':
                        model.set_feature_mask(device,Subset(dataset, train_val_idx))
                    
                    best_model = None
                    best_es_score = {'auc_scores': -np.inf, 'f1': -np.inf, 'recall': -np.inf, 'precision': -np.inf, 'accuracy': -np.inf} 
                    patience_counter = 0
                    
                    for epoch in range(epochs):
                        train_loss, _, _, _ = training_epoch_with_auc_select(
                            model, optimizer, 
                            Subset(dataset, train_val_idx), 
                            args.model_feature_select,
                            args.feature_extract,
                            batch_size, 
                            l1_lambda, 
                            l2_lambda
                        )
                        
                        # Evaluate for early stopping
                        temp_metrics = evaluate_model(model, Subset(dataset,test_val_idx), batch_size)

                        if temp_metrics['auc_scores'] > best_es_score['auc_scores']:
                            for x in temp_metrics:
                                if x in best_es_score:
                                    best_es_score[x] = temp_metrics[x]
                            best_model = deepcopy(model.state_dict())
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter >= early_stopping_patience:
                                print(f"  Early stopping at epoch {epoch}")
                                break
                    for metric in best_es_score:
                        all_results[(l1_lambda,l2_lambda)][metric].append(best_es_score[metric])
                        print(f"   Model {metric} CV score: {best_es_score[metric]:.4f}")
       

    best_param = None
 
    best_results = {'auc_scores': -np.inf, 'f1': -np.inf, 'recall': -np.inf, 'precision':-np.inf, 'accuracy': -np.inf}
    for param in all_results:
        l1_lambda, l2_lambda = param
        print(f"\n\n'l1_lambda': {l1_lambda}, 'l2_lambda': {l2_lambda}")
        mean_score = np.mean(all_results[param][primary_metric])
        if mean_score > best_results[primary_metric]:
            for metric in metrics:
                mean_score = np.mean(all_results[param][metric]) 
                print(f"Mean model {metric} CV score: {mean_score:.4f}")
                best_results[metric] = mean_score
                bet_param = param
    
    return best_results


def evaluate_model(model, dataset, batch_size=None):
    """
    Evaluate binary classification model performance
    
    Parameters:
    -----------
    model : nn.Module
    dataset : Dataset
        Image featuere dataset to evaluate on 
    Returns:
    --------
    dict
        Performance metrics including F1 score, recall, precision, and accuracy
    """

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.eval()

    all_labels = []
    all_probs = []
    loss_fn = torch.nn.BCELoss()
    total_loss = 0.0

    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            x, y = dataset[i]
            pred = model(x.to(device))
            y = y.view(1,1)
            
            # Calculate loss
            loss = loss_fn(pred, y.float().to(device))
            total_loss += loss.item()

            all_labels.append(y.cpu().numpy())
            all_probs.append(pred.cpu().numpy())
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels).flatten()
    all_probs = np.array(all_probs).flatten()
    
    # Search thresholds from 0.0 to 1.0
    thresholds = np.linspace(0, 1, 100)
    best_f1 = 0.0
    best_threshold = 0.5
    best_metrics = {}

    for t in thresholds:
        preds = (all_probs >= t).astype(float)
        f1 = f1_score(all_labels, preds, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_metrics = {
                'loss': total_loss / len(dataset),
                'f1': f1,
                'recall': recall_score(all_labels, preds, zero_division=0),
                'precision': precision_score(all_labels, preds, zero_division=0),
                'accuracy': accuracy_score(all_labels, preds),
                'auc_scores': roc_auc_score(all_labels, all_probs)
            }

    best_metrics['threshold'] = best_threshold
    return best_metrics

def save_results_to_txt(data_dict, cancer_type, output_dir="results"):
    """
    Save results for a specific cancer type to a text file.
    
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the output file path
    output_file = os.path.join(output_dir, f"{cancer_type}_results.txt")
    
    # Write the dictionary to a text file
    with open(output_file, 'w') as f:
        json.dump(data_dict, f, indent=4)
    
    print(f"Results for {cancer_type} saved to {output_file}")

# Tried a RFModel, did not work as well as Linear regession. 
# Code is STALE and has not been updated. Keeping only as a starting block for future trial experiments.
def nested_RFModel(dataset,group_ids,n_outer_folds,input_dim):
    
    dataset_indices = np.arange(len(dataset))
    all_labels = [dataset[i][1] for i in range(len(dataset))]  # Adjust based on your dataset structure
    outer_splits = create_stratified_grouped_cv_splits(dataset_indices, all_labels, group_ids, n_splits=n_outer_folds)
    
    # # Define the parameter grid
    # param_grid = {
    #     'n_estimators': [100, 200, 300, 500],
    #     'max_depth': [None, 10, 20, 30],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    #     'max_features': ['sqrt', 'log2', None]
    # }
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
    }
    all_results = {
        (n_est): {'scores': []} for n_est in  param_grid['n_estimators']
    }

    for outer_fold, (train_val_idx, test_val_idx) in enumerate(outer_splits):
        print(f"\nOuter Fold {outer_fold+1}/{n_outer_folds}")
        for para_typ, param in param_grid.items():
            for n_ext in param:
                rf_model = RandomForestModel_Class(input_dim,n_ext,42)
                rf_model.train_rf_model( Subset(dataset, train_val_idx), True)
                auc_score = rf_model.evaluate_rf_model( Subset(dataset, test_val_idx), True)
                all_results[(n_ext)]['scores'].append(auc_score)
    
    best_parm_score = -np.inf
    best_param = None
    for param in all_results:
        mean_score = np.mean(all_results[param]['scores']) 
        if mean_score > best_parm_score:
            best_param = param
            best_parm_score = mean_score
    print(f"\n n_estimator': {best_param}")
    print(f"Mean model CV score: {mean_score:.4f}\n")


parser = argparse.ArgumentParser(description='train gene status')
parser.add_argument('--cancer_type', type=str,default='BRCA',choices=['BRCA','LUAD','STAD','HNSC','LGG','CESC','LUSC','ESCA','GBM']) 
parser.add_argument('--k_fold_splits', type=int, default=5)
parser.add_argument('--curr_split', type=int, default=24)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--feature_extract', type=str, default= 'uni', choices=['uni','titan'])
parser.add_argument('--data_type', type=str, default= 'True', choices=['True','Synthetic'])
parser.add_argument('--model_feature_select', type=str, default= 'mvavg', choices=['mvavg','global','none'])

parser.add_argument('--base_input_path', type=str, default="/shares/sinha/sadeleye/ecPATH/Data/Training_Data",
                    help="base input path where data files are kept")
parser.add_argument('--feature_input_dir', type=str, default="/shares/sinha/lliu/projects/pre-cancer-image-omics", choices=['/shares/sinha/sadeleye/TITAN_Fets/TCGA_Titan_Fet','/shares/sinha/lliu/projects/pre-cancer-image-omics'],
                    help="base input path where image feaures are kept")

parser.add_argument('--output_path', type=str, default='/shares/sinha/sadeleye/ecPATH_Results/Synthetic_vs_True')
args = parser.parse_args()

if __name__ == '__main__':
    print("Sarting Training.", flush=True)
    # 5 fold cross validation _ set up index
    n_split = args.k_fold_splits
    cancer_type = args.cancer_type 

    # Set to ensure consistancy while testing
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    results_path = os.path.join(args.output_path,
        f"{args.feature_extract}_Test")


    data_dict = defaultdict(dict)
    
    print(f"\nLoading {cancer_type} Data ....", flush=True)
    if args.data_type == 'cpy_numb':
        print("Warning, not fully implemented")
        ecDNA_df = pd.read_csv(args.base_input_path + '/TCGA_tumor_samples_all_cancer_type_ecDNA_and_other_variant_status.csv')
        cpy_df = pd.read_csv(args.base_input_path + '/gene_copy_number/tcga_copy_number_data.tsv', sep='\t')
        auc_scores,feature_names,distro = cpyNumb_dataPrep_DeepPT(cancer_type,cpy_df,ecDNA_df)
    elif args.data_type == "Synthetic":
        # All tumor TCGA info from MLecdna
        ecDNA_df_r = pyreadr.read_r(args.base_input_path + f'/tcga_snp_array_gcap_result2/TCGA_SNP_{cancer_type}_prediction_result.rds')
        ecDNA_df = ecDNA_df_r[None] 
        ecDNA_dataset, groupids = dataPrep_MLecdna(ecDNA_df, cancer_type)
    else:
        # All tumor TCGA info from DeepPT
        ecDNA_df = pd.read_csv(args.base_input_path + '/TCGA_tumor_samples_all_cancer_type_ecDNA_and_other_variant_status.csv')
        ecDNA_dataset, groupids = dataPrep_DeepPT(cancer_type,ecDNA_df)


    print("Training...", flush=True)
    input_dim = 768 if args.feature_extract == "titan" else 1024 # Uni or Titan embedding Size

    # This is a hard fix for cancer type CESC with True data. Sample size is 70 with 20% postive. With a 5 fold split w/o randomness, and trying to keep % of postive with each fold
    #  this give us one fold with all negatives labels. We do 4 folds instead. 
    if cancer_type == 'CESC' and args.data_type == "True":
        n_split = 4
    else:
        n_split = args.k_fold_splits

    results = nested_cv_with_regularization_grouped(ecDNA_dataset,input_dim,groupids, epochs=32,n_outer_folds=n_split)
    data_dict[args.data_type][cancer_type] = results
    
    save_results_to_txt(data_dict,cancer_type,results_path)
          


    