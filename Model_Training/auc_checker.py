import numpy as np
import pandas as pd
import argparse
import pyreadr
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset
import os
import h5py
import mygene

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
    
def feature_dataPrep_DeepPT(cancer_type,ecDNA_df):
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

def feature_dataPrep_MLecdna(ecDNA_df, cancer_type):
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

def features_auc_scores(dataset):
    # Initialize arrays to collect mean features and labels
    mean_features = []
    all_labels = []
    
    # Process each sample
    for features, label in dataset:
        if args.feature_extract == 'titan':
            mean_features.append(features)
        else:
            # Handle different possible data formats
            if isinstance(features, torch.Tensor):
                # Average across the tiles dimension (dim 0)
                sample_mean = features.mean(dim=0).cpu().numpy()
            else:
                # If numpy array or list
                sample_mean = np.mean(features, axis=0)
            # Append to our collections
            mean_features.append(sample_mean)
            
        # Convert label to appropriate format
        if isinstance(label, torch.Tensor):
            label_value = label.item() if label.numel() == 1 else label.cpu().numpy()
        else:
            label_value = label
            
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
                auc = roc_auc_score(labels_array, features_stack[:, i])
                if auc < 0.5:
                    auc = 1 - auc
                auc_scores[i] = auc
            else:
                auc_scores[i] = 0.5  # Default AUC for non-predictive features
        except Exception as e:
            print(f"Error calculating AUC for feature {i}: {e}")
            auc_scores.append(0.5)
    
    # Return array of AUC scores
    return auc_scores

def cpyNumb_dataPrep_MLecdna(cancer_type, cpData_df, ecDNA_df):
    # Sample and gene status data
    sample_ecDNA_status_df = ecDNA_df.groupby('sample').agg({
            'gene_class': lambda x: '1' if 'circular' in set(x) else '0'
            }).reset_index()
    # sample_ecDNA_status_df = sample_ecDNA_status_df.rename(columns={'gene_class': 'ecDNA_status'}) # rename to ecDna_status
    ecDNA_df = sample_ecDNA_status_df.rename(columns={'gene_class': 'ecDNA_status'}) # rename to ecDna_status
    ecDNA_samples = sample_ecDNA_status_df['sample'].tolist()



    cpyNumb_samples = cpData_df.columns.tolist()[1:]
    cp_transposed = cpData_df.set_index('Sample').T


    ecDNA_df_type = ecDNA_df[
        (ecDNA_df["sample"].isin(cpyNumb_samples))
    ].reset_index(drop=True)
    ecDNA_df_type['sample_id'] = ecDNA_df_type["sample"]
    ecDNA_df_type = ecDNA_df_type.set_index('sample_id')

    merged_df_inner = cp_transposed.join(
        ecDNA_df_type,
        how='inner'
    )

    # Not needed, all cols have variance. 
    no_variance_cols = [col for col in merged_df_inner.columns 
                      if merged_df_inner[col].nunique() == 1]


    non_feature_cols = ['sample', 'ecDNA_status']
    feature_cols = [col for col in merged_df_inner.columns if col not in non_feature_cols or col in no_variance_cols]
    X = merged_df_inner[feature_cols].values
    y = merged_df_inner['ecDNA_status'].astype(int).values
    auc_scores = []
    feature_names = []

    # Stack all mean feature vectors
    for i in range(X.shape[1]):
        try:
            # Calculate AUC
            auc = roc_auc_score(y, X[:, i])
            
            # If AUC < 0.5, the relationship is inverse, so use 1-AUC
            if auc < 0.5:
                auc = 1 - auc
                
            auc_scores.append(auc)
            feature_names.append(feature_cols[i])
        except Exception as e:
            print(f"Error calculating AUC for feature {feature_cols[i]}: {e}")
    print(f"Sum of Y:{sum(y)}")
    print(f"Lenght of Y:{len(y)}")
    return auc_scores, feature_names, (X.shape[0],(sum(y)/len(y))*100)

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
    X = merged_df_inner[feature_cols].values
    y = merged_df_inner['ecDNA_status'].values

    auc_scores = []
    feature_names = []

    # Stack all mean feature vectors
    for i in range(X.shape[1]):
        try:
            # Calculate AUC
            auc = roc_auc_score(y, X[:, i])
            
            # If AUC < 0.5, the relationship is inverse, so use 1-AUC
            if auc < 0.5:
                auc = 1 - auc
                
            auc_scores.append(auc)
            feature_names.append(feature_cols[i])
        except Exception as e:
            print(f"Error calculating AUC for feature {feature_cols[i]}: {e}")
    print(f"Sum of Y:{sum(y)}")
    print(f"Lenght of Y:{len(y)}")
    return auc_scores, feature_names, (X.shape[0],(sum(y)/len(y))*100)


def convert_ftNames(feature_names):
    gene_names = []
    mg = mygene.MyGeneInfo()
    
    for name in feature_names:
        result = mg.query('symbol:' + name, species='human', fields='ensembl.gene')
        ensembl_data = result['hits'][0].get('ensembl', {})
        if isinstance(ensembl_data, dict):
            gene_names.append(ensembl_data.get('gene', 'Not found'))
        elif isinstance(ensembl_data, list) and len(ensembl_data) > 0:
            gene_names.append(ensembl_data[0].get('gene', 'Not found'))

    return gene_names

parser = argparse.ArgumentParser(description='train gene status')
parser.add_argument('--cancer_type', type=str,default='BRCA',choices=['BRCA','LUAD','STAD','HNSC','LGG','CESC','LUSC','ESCA','GBM']) 

parser.add_argument('--feature_extract', type=str, default= 'uni', choices=['uni','resnet','titan'])
parser.add_argument('--data_type', type=str, default= 'cpy_numb', choices=['True','Synthetic', 'cpy_numb_True','cpy_numb_Synthetic'])

parser.add_argument('--base_input_path', type=str, default="/shares/sinha/sadeleye/ecPATH/Data/Training_Data",
                    help="base input path where data files are kept")

parser.add_argument('--feature_input_dir', type=str, default="/shares/sinha/lliu/projects/pre-cancer-image-omics", choices=['/shares/sinha/sadeleye/TITAN_Fets/TCGA_Titan_Fet','/shares/sinha/lliu/projects/pre-cancer-image-omics'],
                    help="base input path where image feaures are kept")

parser.add_argument('--output_path', type=str, default='/shares/sinha/sadeleye/ecPATH_Results')
args = parser.parse_args()

if __name__ == '__main__':
    # Use this to ensure print statements are immediately flushed
    print("Sarting Training.", flush=True)


    # cancer_type = args.cancer_type 

    cancer_types = ['ESCA','BRCA','LUAD','STAD','HNSC','LGG','CESC','LUSC','GBM']
    for cancer_type in cancer_types:
        args.cancer_type = cancer_type
        print(f"Loading {cancer_type} Data ....", flush=True)
        auc_scores = None 
        feature_names = None
        if args.data_type == 'True':
            ecDNA_df = pd.read_csv(args.base_input_path + '/TCGA_tumor_samples_all_cancer_type_ecDNA_and_other_variant_status.csv')
            ecDNA_dataset, groupids = feature_dataPrep_DeepPT(cancer_type,ecDNA_df)
            auc_scores = features_auc_scores(ecDNA_dataset)
        if args.data_type == "Synthetic":
            ecDNA_df_r = pyreadr.read_r(args.base_input_path + f'/tcga_snp_array_gcap_result2/TCGA_SNP_{cancer_type}_prediction_result.rds')
            ecDNA_df = ecDNA_df_r[None] 
            ecDNA_dataset, groupids = feature_dataPrep_MLecdna(ecDNA_df,cancer_type)
            auc_scores = features_auc_scores(ecDNA_dataset)
        if args.data_type == 'cpy_numb_Synthetic':
            cpy_df = pd.read_csv(args.base_input_path + '/gene_copy_number/tcga_copy_number_data.tsv', sep='\t')
            ecDNA_df_r = pyreadr.read_r(args.base_input_path + f'/tcga_snp_array_gcap_result2/TCGA_SNP_{cancer_type}_prediction_result.rds')
            ecDNA_df = ecDNA_df_r[None] 
            auc_scores,feature_names,distro = cpyNumb_dataPrep_MLecdna(cancer_type,cpy_df,ecDNA_df)
        if args.data_type == 'cpy_numb_True':
            cpy_df = pd.read_csv(args.base_input_path + '/gene_copy_number/tcga_copy_number_data.tsv', sep='\t')
            ecDNA_df_r = pyreadr.read_r(args.base_input_path + f'/tcga_snp_array_gcap_result2/TCGA_SNP_{cancer_type}_prediction_result.rds')
            ecDNA_df = pd.read_csv(args.base_input_path + '/TCGA_tumor_samples_all_cancer_type_ecDNA_and_other_variant_status.csv')
            auc_scores,feature_names,distro = cpyNumb_dataPrep_DeepPT(cancer_type,cpy_df,ecDNA_df)
            


        top_indices = np.argsort(auc_scores)[::-1][:10]

        with open(f"{cancer_type}_top_{args.data_type}_features_auc_Synthetic.txt", "w") as f:
            f.write("Top 10 features by AUC:\n")
            f.write(f"{distro[0]} Samples with {distro[1]:2f}% Postive\n")
            for idx in top_indices:
                try:
                    gene_name = convert_ftNames([feature_names[idx]])[0]
                except Exception as e:
                    print(f"Error converting Featurename {feature_names[idx]}: {e}")
                    gene_name = feature_names[idx]
                f.write(f"Feature {gene_name}: AUC = {auc_scores[idx]:.4f}\n")
                print(f"Feature {gene_name}: AUC = {auc_scores[idx]:.4f}")
            # Write the overall mean AUC
            f.write(f"{np.mean(auc_scores)}\n")
            print(np.mean(auc_scores))
        