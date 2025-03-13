import warnings

import numpy as np
import pandas as pd
import torch

np.warnings = warnings
import os
import pickle

np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

import os
import sys
import time

import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data import Dataset, Subset
import argparse
import pyreadr
from ecdna_label_residual_model import training_epoch, EcDNATileClassifier
from utils import Feature_Dataset, get_detailed_metrics, create_grouped_cv_splits
from copy import deepcopy
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, roc_auc_score





def dataPrep_MLecdna(ecDNA_df, cancer_type, results_path, n_split, sample_split_path,):
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
        feature_extract_dict = {"uni":"-uni.npy", "resnet":".npy"}
        feature_extinson = feature_extract_dict[args.feature_extract]
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

    return Feature_Dataset(file_paths,targets), group_ids


def dataPrep_DeepPt(ecDNA_df , cur_results_len, results_path, feature_extractor, ):
    pass    


def nested_cv_with_regularization_grouped(dataset, input_dim, group_ids, n_outer_folds=5, n_inner_folds=5, 
                                         batch_size=32, epochs=30, hidden_dim=256, 
                                         learning_rate=0.001, early_stopping_patience=5,
                                         primary_metric='auc'):
    """
    Parameters:
    -----------
    dataset : Dataset
        Full Image feature dataset
    input_dim : int
        Input dimension for the model - preset for UNI now (testing)
    group_ids : list
        Unique group ids (e.g., patient IDs) for each sample
    n_outer_folds : int
        Number of folds for outer CV
    n_inner_folds : int
        Number of folds for inner CV
    batch_size : int
        Batch size for training
    epochs : int
        Maximum number of epochs
    hidden_dim : int
        Hidden dimension for the model
    learning_rate : float
        Learning rate for optimizer
    early_stopping_patience : int
        Number of epochs to wait before early stopping
    primary_metric : str
        Metric to use for model selection ('f1', 'accuracy', 'auc')
        
    Returns:
    --------
    dict
        Results of nested CV, including best regularization type and hyperparameters
    """
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    
    # Define regularization types and parameters to try
    regularization_params = {
        'L1 only': {'l1_values': [0.0001, 0.0005, 0.001, 0.005, 0.01], 'l2_values': [0.0]},
        'L2 only': {'l1_values': [0.0], 'l2_values': [0.0001, 0.0005, 0.001, 0.005, 0.01]},
        'Elastic Net': {'l1_values': [0.0001, 0.001, 0.01], 'l2_values': [0.0001, 0.001, 0.01]}
    }
    

    dataset_indices = np.arange(len(dataset))
    
    # Create OUTER CV splits respecting group constraints
    outer_splits = create_grouped_cv_splits(dataset_indices, group_ids, n_splits=n_outer_folds)
    
    # Results tracking
    all_results = {reg_type: {'scores': [], 'detailed_metrics': [], 'best_params': []} for reg_type in regularization_params.keys()}
    
    # Outer CV loop
    for outer_fold, (train_val_idx, test_idx) in enumerate(outer_splits):
        print(f"\nOuter Fold {outer_fold+1}/{n_outer_folds}")
        
        # Get group IDs for the train_val samples
        train_val_group_ids = [group_ids[i] for i in train_val_idx]
        
        # Create INNER CV splits with groups constraints
        inner_splits = create_grouped_cv_splits(train_val_idx, train_val_group_ids, n_splits=n_inner_folds)
        
        # For each regularization type
        for reg_type, params in regularization_params.items():
            print(f"\nEvaluating {reg_type} regularization")
            
            best_params = None
            best_score = -np.inf
            
            # Grid search over regularization parameters
            for l1_lambda in params['l1_values']:
                for l2_lambda in params['l2_values']:
                    if l1_lambda == 0.0 and l2_lambda == 0.0:
                        continue  # Skip no regularization case
                    
                    print(f"  Testing L1={l1_lambda}, L2={l2_lambda}")
                    
                    # Inner CV for this parameter combination
                    inner_scores = []
                    
                    for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_splits):
                        # Create model
                        model = EcDNATileClassifier(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                        
                        # Train model with current regularization parameters
                        best_val_score = -np.inf
                        patience_counter = 0
                        
                        for epoch in range(epochs):
                            train_loss, _, _, _ = training_epoch(
                                model, optimizer, 
                                Subset(dataset, inner_train_idx), 
                                batch_size, l1_lambda, l2_lambda
                            )
                            
                            # Evaluate on validation set
                            val_score, vald_loss = evaluate_model(model, Subset(dataset, inner_val_idx), batch_size)
                            
                            # Early stopping check
                            if val_score > best_val_score:
                                best_val_score = val_score
                                patience_counter = 0
                            else:
                                patience_counter += 1
                                if patience_counter >= early_stopping_patience:
                                    print(f"    Early stopping at epoch {epoch}")
                                    break
                        
                        inner_scores.append(best_val_score)
                    
                    # Average score across inner folds
                    mean_score = np.mean(inner_scores)
                    print(f"    Mean inner CV score: {mean_score:.4f}")
                    
                    # Update best parameters if better
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = {'l1_lambda': l1_lambda, 'l2_lambda': l2_lambda}
            
            print(f"  Best parameters for {reg_type}: {best_params}, score: {best_score:.4f}")
            
            # Extract a small, group-conscious validation set for early stopping
            # We need to identify a subset of groups to hold out
            unique_groups = np.unique(train_val_group_ids)
            np.random.shuffle(unique_groups)
            es_groups = unique_groups[:max(1, len(unique_groups) // 10)]  # 10% of groups
            
            # Find indices corresponding to early stopping groups
            early_stop_mask = np.isin(train_val_group_ids, es_groups)
            early_stop_indices = train_val_idx[early_stop_mask]
            final_train_indices = train_val_idx[~early_stop_mask]
            
            # Train a model with best parameters on all train_val data (except early stopping subset)
            model = EcDNATileClassifier(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            best_model = None
            best_es_score = -np.inf
            patience_counter = 0
            
            for epoch in range(epochs):
                train_loss, _, _, _ = training_epoch(
                    model, optimizer, 
                    Subset(dataset, final_train_indices), 
                    batch_size, 
                    best_params['l1_lambda'], 
                    best_params['l2_lambda']
                )
                
                # Evaluate for early stopping
                es_score, _ = evaluate_model(model, Subset(dataset, early_stop_indices), batch_size)
                
                if es_score > best_es_score:
                    best_es_score = es_score
                    best_model = deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"  Early stopping at epoch {epoch}")
                        break
            
            # Load best model
            model.load_state_dict(best_model)
            
            # Evaluate on test set (get primary metric for comparison)
            test_score, _ = evaluate_model(model, Subset(dataset, test_idx), batch_size)
            print(f"  Test score for {reg_type}: {test_score:.4f}")
            
            # Get detailed metrics for final reporting
            detailed_metrics = get_detailed_metrics(model, Subset(dataset, test_idx), batch_size)
            
            # Store results
            all_results[reg_type]['scores'].append(test_score)
            all_results[reg_type]['detailed_metrics'].append(detailed_metrics)
            all_results[reg_type]['best_params'].append(best_params)
    
    # Calculate mean and std for each regularization type
    summary = {}
    for reg_type in regularization_params.keys():
        scores = all_results[reg_type]['scores']
        detailed_metrics_list = all_results[reg_type]['detailed_metrics']
        
        # Aggregate detailed metrics
        aggregated_metrics = {}
        for metric in detailed_metrics_list[0].keys():
            values = [d[metric] for d in detailed_metrics_list]
            aggregated_metrics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
        
        summary[reg_type] = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'best_params': all_results[reg_type]['best_params'],
            'detailed_metrics': aggregated_metrics
        }
    
    # Determine best regularization type
    best_reg_type = max(summary.keys(), key=lambda k: summary[k]['mean_score'])
    
    # Find most common best parameters
    all_best_params = all_results[best_reg_type]['best_params']
    param_counts = {}
    for params in all_best_params:
        param_key = f"L1={params['l1_lambda']}, L2={params['l2_lambda']}"
        if param_key not in param_counts:
            param_counts[param_key] = 0
        param_counts[param_key] += 1
    
    most_common_params = max(param_counts.keys(), key=lambda k: param_counts[k])
    
    # Return results
    return {
        'summary': summary,
        'best_regularization': best_reg_type,
        'mean_score': summary[best_reg_type]['mean_score'],
        'std_score': summary[best_reg_type]['std_score'],
        'most_common_params': most_common_params,
        'detailed_results': all_results
    }

def evaluate_model(model, dataset,batch_size=None):
    """
    Evaluate binary classification model performance by processing each sample individually
    
    Parameters:
    -----------
    model : nn.Module
    dataset : Dataset
        image feature dataset to evaluate on
        
    Returns:
    --------
    float
        AUC score (primary metric for model selection)
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.eval()
    
    all_labels = []
    all_probs = []
    loss_fn = torch.nn.BCELoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for i in range(len(dataset)):
            x, y = dataset[i]
            pred = model(x.to(device))
            y = y.view(1,1) # change y shave to be 1 x 1 vector
            
            # Calculate loss
            loss = loss_fn(pred, y.float().to(device))
            total_loss += loss.item()

            # Store results (raw probabilities for AUC calculation)
            all_labels.append(y.cpu().numpy())
            all_probs.append(pred.cpu().numpy())
    
    # Convert lists to numpy arrays and flatten
    all_labels = np.array(all_labels).flatten()
    all_probs = np.array(all_probs).flatten()
    

    # Calculate average loss
    avg_loss = total_loss / len(dataset)

    # Check if we have both classes in the dataset
    unique_labels = np.unique(all_labels)
    if len(unique_labels) < 2:
        print(f"Warning: Only one class present in evaluation set ({unique_labels[0]}). Returning 0.5 AUC.")
        return 0.5
    
    # Calculate AUC score
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception as e:
        print(f"Error calculating AUC: {e}")
        print(f"Unique labels: {unique_labels}, Label counts: {np.bincount(all_labels.astype(int))}")
        # Fallback to accuracy if AUC calculation fails
        binary_preds = (all_probs >= 0.5).astype(float)
        return accuracy_score(all_labels, binary_preds)
    
    return auc, avg_loss

def evaluate_model_f1(model, dataset, batch_size=None):
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
    import numpy as np
    import torch
    from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    loss_fn = torch.nn.BCELoss()
    total_loss = 0.0

    with torch.no_grad():
        for i in range(len(dataset)):
            x, y = dataset[i]
            pred = model(x.to(device))
            
            # Calculate loss
            loss = loss_fn(pred, y.float().to(device))
            total_loss += loss.item()
            
            # Convert to binary predictions
            binary_pred = (pred.cpu() >= 0.5).float()
            
            # Store results
            all_labels.append(y.cpu().numpy())
            all_preds.append(binary_pred.numpy())
            all_probs.append(pred.cpu().numpy())
    
    # Convert lists to numpy arrays
    all_labels = np.array(all_labels).flatten()
    all_preds = np.array(all_preds).flatten()
    all_probs = np.array(all_probs).flatten()
    
    # Calculate average loss
    avg_loss = total_loss / len(dataset)
    
    # Calculate metrics
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return {
        'loss': avg_loss,
        'f1': f1,
        'recall': recall, 
        'precision': precision,
        'accuracy': accuracy
    }



parser = argparse.ArgumentParser(description='train gene status')
parser.add_argument('--cancer_type', type=str,default='BRCA',choices=['BRCA','LUAD','STAD','HNSC','LGG','CESC','LUSC','ESCA','GBM']) 
parser.add_argument('--ml_methods', type=str, default='LR', choices=['LR', 'GB', 'SVM', 'RF'], 
                    help= "logistic regression, gradient boosting, SVM, Random Forest")

parser.add_argument('--k_fold_splits', type=int, default=5)
parser.add_argument('--curr_split', type=int, default=24)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--feature_extract', type=str, default= 'uni', choices=['uni','resnet'])
parser.add_argument('--data_type', type=str, default= 'MLecdna', choices=['DeepPt','MLecdna'])

parser.add_argument('--base_input_path', type=str, default="/shares/sinha/sadeleye/ecPATH/Data/Training_Data",
                    help="base input path where data files are kept")
parser.add_argument('--feature_input_dir', type=str, default="/shares/sinha/lliu/projects/pre-cancer-image-omics",
                    help="base input path where image feaures are kept")

parser.add_argument('--output_path', type=str, default='/shares/sinha/sadeleye/ecPATH_Results')
args = parser.parse_args()

if __name__ == '__main__':
    # Use this to ensure print statements are immediately flushed
    print("Sarting Training.", flush=True)
    # 5 fold cross validation _ set up index
    n_split = args.k_fold_splits
    cancer_type = args.cancer_type 
    # cur_split_selection = args.curr_split


    results_path = os.path.join(args.output_path,
        "ecdna_LabelResult_Res",
        cancer_type,
        f"{n_split}_folds_cur_{args.feature_extract}",
    )

    if not os.path.exists(results_path):
        print("output path created: ", results_path)
        os.makedirs(results_path)
    # cur_results_len = len(os.listdir(results_path))



    sample_split_path = os.path.join(
        args.base_input_path,
        "Sample_splits",
        f"sample_split_{n_split}_{n_split}_fold_{cancer_type}_{args.data_type}.pkl",
    )
    if not os.path.exists(sample_split_path):
        os.makedirs(sample_split_path)

    print("Loading Data ....", flush=True)
    if args.data_type == 'DeepPt':
        ecDNA_df = pd.read_csv(args.base_input_path + '/TCGA_eCDNA/data/TCGA_tumor_samples_all_cancer_type_ecDNA_and_other_variant_status.csv')
        merged_ecDNA_df = dataPrep_DeepPt(ecDNA_df, cancer_type, results_path, args.feature_extract)
    else:
        # All tumor TCGA info from MLecdna
        ecDNA_df_r = pyreadr.read_r(args.base_input_path + f'/tcga_snp_array_gcap_result2/TCGA_SNP_{cancer_type}_prediction_result.rds')
        ecDNA_df = ecDNA_df_r[None] 
        ecDNA_dataset, groupids = dataPrep_MLecdna(ecDNA_df, cancer_type, results_path,n_split,sample_split_path)
    
    print("Training...", flush=True)
    results = nested_cv_with_regularization_grouped(ecDNA_dataset,1024,groupids)
    print(results, flush=True)

    