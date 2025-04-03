
import numpy as np
import pandas as pd
import argparse
import pyreadr
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset
import os
import h5py

class Feature_Dataset(Dataset):
    def __init__(self,filepaths, targets,ext):
        """
        Args:
        file_path (string): Path to .npy file containing slide feature data.
        """
        self.features = []
        if ext == '.h5':
            for temp_feature in filepaths:
                with h5py.File(temp_feature, "r") as file:
                    feature = file['embedding'][:]
                    self.features.append(feature.astype(np.float32))
        else:
            self.features = [np.load(temp_feature).astype(np.float32) for temp_feature in filepaths]
        self.targets = np.array(targets, dtype=np.float32)
        self.length = len(self.features)

    def __len__(self):
        return self.length

    def __getitem__(self,idx):
        sample = torch.Tensor(self.features[idx]).float()
        target = torch.tensor([self.targets[idx]], dtype=torch.float)
        return sample, target
    
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
        if feature_extinson == ".h5":
            temp_feature_path = os.path.join(args.feature_input_dir,
                                             row["filename"].split(".")[0] + feature_extinson,)
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
    print(f"{(sum(targets)/len(targets))*100:.2f}% of samples are Postive")
    return Feature_Dataset(file_paths,targets,feature_extinson), group_ids



def features_auc_scores(dataset):

    # Initialize arrays to collect mean features and labels
    mean_features = []
    all_labels = []
    
    # Process each sample
    for features, label in dataset:
        # Handle different possible data formats
        # if isinstance(features, torch.Tensor):
        #     # Average across the tiles dimension (dim 0)
        #     sample_mean = features.mean(dim=0).cpu().numpy()
        # else:
        #     # If numpy array or list
        #     sample_mean = np.mean(features, axis=0)
        mean_features.append(features)
        
        # Convert label to appropriate format
        if isinstance(label, torch.Tensor):
            label_value = label.item() if label.numel() == 1 else label.cpu().numpy()
        else:
            label_value = label
            
        # Append to our collections
        # mean_features.append(sample_mean)
        all_labels.append(label_value)
    
    # Stack all mean feature vectors
    features_stack = np.vstack(mean_features)  # Shape: (n_samples, 1024)
    labels_array = np.array(all_labels)        # Shape: (n_samples,)
    
    # Calculate AUC for each feature
    auc_scores = np.zeros(features_stack.shape[1])
    
    for i in range(features_stack.shape[1]):
        try:
            # Handle possible errors (e.g., single class in labels or constant feature)
            if len(np.unique(labels_array)) > 1 and len(np.unique(features_stack[:, i])) > 1:
                auc_scores[i] = roc_auc_score(labels_array, features_stack[:, i])
            else:
                auc_scores[i] = 0.5  # Default AUC for non-predictive features
        except Exception as e:
            print(f"Error calculating AUC for feature {i}: {e}")
            auc_scores[i] = 0.5
    
    # Return array of AUC scores
    return auc_scores



parser = argparse.ArgumentParser(description='train gene status')
parser.add_argument('--cancer_type', type=str,default='BRCA',choices=['BRCA','LUAD','STAD','HNSC','LGG','CESC','LUSC','ESCA','GBM']) 
parser.add_argument('--ml_methods', type=str, default='LR', choices=['LR', 'GB', 'SVM', 'RF'], 
                    help= "logistic regression, gradient boosting, SVM, Random Forest")

parser.add_argument('--k_fold_splits', type=int, default=5)
parser.add_argument('--curr_split', type=int, default=24)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--feature_extract', type=str, default= 'titan', choices=['uni','resnet','titan'])
parser.add_argument('--data_type', type=str, default= 'DeepPt', choices=['DeepPt','MLecdna'])

parser.add_argument('--base_input_path', type=str, default="/shares/sinha/sadeleye/ecPATH/Data/Training_Data",
                    help="base input path where data files are kept")

parser.add_argument('--feature_input_dir', type=str, default="/shares/sinha/sadeleye/TITAN_Fets/TCGA_Titan_Fet", choices=['/shares/sinha/sadeleye/TITAN_Fets/TCGA_Titan_Fet','/shares/sinha/lliu/projects/pre-cancer-image-omics'],
                    help="base input path where image feaures are kept")

parser.add_argument('--output_path', type=str, default='/shares/sinha/sadeleye/ecPATH_Results')
args = parser.parse_args()

if __name__ == '__main__':
    # Use this to ensure print statements are immediately flushed
    print("Sarting Training.", flush=True)
    # 5 fold cross validation _ set up index
    n_split = args.k_fold_splits
    cancer_type = args.cancer_type 

    cancer_types = ['BRCA','LUAD','STAD','HNSC','LGG','CESC','LUSC','ESCA','GBM']
    for cancer_type in cancer_types:
        # cur_split_selection = args.curr_split
        print("Loading Data ....", flush=True)
        if args.data_type == 'DeepPt':
            ecDNA_df = pd.read_csv(args.base_input_path + '/TCGA_tumor_samples_all_cancer_type_ecDNA_and_other_variant_status.csv')
            ecDNA_dataset, groupids = dataPrep_DeepPT(cancer_type,ecDNA_df)

        auc_scores = features_auc_scores(ecDNA_dataset)
        top_indices = np.argsort(auc_scores)[::-1][:10]

        with open(f"{cancer_type}_top_Titan_features_auc.txt", "w") as f:
            f.write("Top 10 features by AUC:\n")
            for idx in top_indices:
                f.write(f"Feature {idx}: AUC = {auc_scores[idx]:.4f}\n")
                print(f"Feature {idx}: AUC = {auc_scores[idx]:.4f}")
            # Write the overall mean AUC
            f.write(f"{np.mean(auc_scores)}\n")
            print(np.mean(auc_scores))
        