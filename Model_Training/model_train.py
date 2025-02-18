import warnings

import numpy as np
import pandas as pd

np.warnings = warnings
import os
import pickle

from sklearn.model_selection import KFold, train_test_split

np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

import os
import sys
import time

import numpy as np
import pandas as pd
# from model_MLP import *
from model import *
from torch.utils.data import ConcatDataset, DataLoader
from utils import *
from torch.utils.data import Dataset, Subset
import argparse


def dataPrep(cur_results_len, sample_split_path, results_path, feature_extractor):
    if cur_results_len == 9:
        print("has done")
    else:
        # different strategy for collecting features
        # 1. muliple slides for multiple expression profile (dup)
        # read in rna-seq as guide to match patient id
        # ranseq meta guide
        # All GENE expression true lables 
        ranseq_meta_guide = pd.read_csv(
            os.path.join(
                BASE,
                "rawData",
                "htseq_fpkm",
                cur_type,
                f"TCGA-{cur_type}.htseq_fpkm_t_norm.tsv",
            ),
            sep=" ",
        )

        # reading in meta as guide
        # Reading in TCGA slide lables by tumor type and making sure id matches a gene expression we have.  
        slides_meta_guide_df = pd.read_csv(
            os.path.join(BASE, "all_slides_TCGA_filtered.tsv"), sep="\t"
        )
        slides_meta_guide_df_type = slides_meta_guide_df[
            (slides_meta_guide_df["source_abrev"] == cur_type)
            & (
                slides_meta_guide_df["slide_submitter_id"].isin(
                    ranseq_meta_guide.index.unique()
                )
            )
        ].reset_index(drop=True)
        #
        # Merged TCGA slide labe info with Gene expression data, keeping only what we have. 
        # slide_submitter_id ENSGOOO ENSGO... ENSGo...
        rna_expr_merged = pd.merge(
            slides_meta_guide_df_type[["slide_submitter_id"]],
            ranseq_meta_guide,
            left_on="slide_submitter_id",
            right_index=True,
        ).reset_index(drop=True)

        # standardizing gene experssion by column 
        mad_values = rna_expr_merged.iloc[:, 1:].apply(mad)
        # column_variances = rna_expr_merged.iloc[:, 1:].std(axis=0)
        
        # eleminate genese with not enough variance 
        columns_to_keep1 = mad_values[mad_values > 0].index
        # columns_to_keep2 = column_variances[column_variances < 0.3].index

        # new dataframe with filterd columns 
        filtered_df_median = rna_expr_merged[
            [rna_expr_merged.columns[0]] + list(columns_to_keep1)
        ]
        # filtered_df_var = rna_expr_merged[[rna_expr_merged.columns[0]] + list(columns_to_keep2)]

        # Genes of columns we now want to predict. 
        genes_to_predict = filtered_df_median.columns[
            1:
        ]  # genes_to_predict = rna_expr_merged.columns[1:]

        print("total number of genes to predict: ", len(genes_to_predict))

        all_unique_patients_slides = slides_meta_guide_df_type["case_submitter_id"].unique()
        print("number of unique patients: ", len(all_unique_patients_slides))

        if not os.path.exists(sample_split_path):
            print(sample_split_path)
            #
            # Initialize lists to store patient IDs for train, validation, and test sets
            patient_id_train_all = []
            patient_id_valid_all = []
            patient_id_test_all = []

            # KFold instance for splitting data
            KFold_split = KFold(n_splits=n_split, shuffle=True, random_state=random_seed)

            # First KFold split to separate train/validation and test sets
            for patient_idx_train_valid_temp, patient_idx_test_temp in KFold_split.split(
                all_unique_patients_slides
            ):
                patient_id_train_valid_temp = all_unique_patients_slides[
                    patient_idx_train_valid_temp
                ]
                # Second KFold split to separate train and validation sets within the train/validation set
                for patient_idx_train_temp, patient_idx_valid_temp in KFold_split.split(
                    patient_id_train_valid_temp
                ):

                    # Append the patient IDs for the train and validation sets
                    patient_id_train_all.append(
                        patient_id_train_valid_temp[patient_idx_train_temp]
                    )
                    patient_id_valid_all.append(
                        patient_id_train_valid_temp[patient_idx_valid_temp]
                    )
                    patient_id_test_all.append(
                        all_unique_patients_slides[patient_idx_test_temp]
                    )

                with open(sample_split_path, "wb") as file:
                    pickle.dump(
                        (patient_id_train_all, patient_id_valid_all, patient_id_test_all),
                        file,
                    )
        else:
            with open(sample_split_path, "rb") as file:
                patient_id_train_all, patient_id_valid_all, patient_id_test_all = (
                    pickle.load(file)
                )

        patient_id_train_cur = patient_id_train_all[cur_split_selection]
        patient_id_valid_cur = patient_id_valid_all[cur_split_selection]
        patient_id_test_cur = patient_id_test_all[cur_split_selection]
        print("patient_id_train_cur: ", len(patient_id_train_cur))
        print("patient_id_valid_cur: ", len(patient_id_valid_cur))
        print("patient_id_test_cur: ", len(patient_id_test_cur))
        
        all_feature = []
        all_target = []
        feature_not_fount = []
        slides_id_train_cur = []
        slides_id_valid_cur = []
        slides_id_test_cur = []
        pred_expr_row_names = []
        idx_off_set = 0

        for idx, row in slides_meta_guide_df_type.iterrows():
            temp_slide_id = row["slide_submitter_id"]
            feature_extract_dict = {"uni":"-uni.npy", "resnet":".npy"}
            feature_file_ext = feature_extract_dict[feature_extractor]
            temp_feature_path = os.path.join(
                BASE,
                "rawData",
                "slides",
                cur_type,
                row["id"],
                "_features",
                row["filename"].replace(
                    ".svs", feature_file_ext
                ),  # TODO: ResNet50 (.npy) modify for UNI (-uni.npy),
            )

            ### compose compilation of features and targets
            try:
                # attach feature
                temp_feature = np.load(temp_feature_path)
                all_feature.append((row["slide_id"], temp_feature))
                # attach gene expr
                # temp = np.delete(rna_expr_merged[rna_expr_merged['slide_submitter_id'] == temp_slide_id].iloc[0:1].values, 0, axis=1) #list(range(19468))
                filtered_df = rna_expr_merged[
                    rna_expr_merged["slide_submitter_id"] == temp_slide_id
                ]
                first_row_df = filtered_df.iloc[0:1]
                first_row_array = first_row_df[genes_to_predict].values
                modified_array = np.delete(first_row_array, 0, axis=1).astype(np.float32)

                all_target.append(np.squeeze(modified_array))

                ### compose slides indexes
                temp_patient_id = row["case_submitter_id"]

                if temp_patient_id in patient_id_train_cur:
                    slides_id_train_cur.append(idx - idx_off_set)
                elif temp_patient_id in patient_id_valid_cur:
                    slides_id_valid_cur.append(idx - idx_off_set)
                elif temp_patient_id in patient_id_test_cur:
                    slides_id_test_cur.append(idx - idx_off_set)
                    pred_expr_row_names.append(temp_slide_id)

            except FileNotFoundError:
                idx_off_set += 1
                feature_not_fount.append(temp_feature_path)

        dataset = slide_target_dataset(all_feature, all_target)
        train_set = Subset(dataset, slides_id_train_cur)
        valid_set = Subset(dataset, slides_id_valid_cur)
        test_set = Subset(dataset, slides_id_test_cur)

        ### output col / row names
        # genes_to_predict
        genes_to_predict_df = pd.DataFrame(genes_to_predict, columns=["genes_to_predict"])

        genes_to_predict_df.to_csv(
            os.path.join(results_path, "genes_to_predict.txt"), index=False
        )
        print(
            f"genes_to_predict_df outup {os.path.join(results_path,'genes_to_predict.txt')}"
        )

        # pred_expr_row_names
        pred_expr_row_names_df = pd.DataFrame(
            pred_expr_row_names, columns=["pred_expr_row_names"]
        )

        pred_expr_row_names_df.to_csv(
            os.path.join(results_path, "pred_expr_row_names.txt"), index=False
        )

        print(
            f"pred_expr_row_names outup {os.path.join(results_path,'pred_expr_row_names.txt')}"
        )
        return train_set, valid_set, test_set, genes_to_predict, all_target
        
def train(all_target, train_set, valid_set):
    ##
    n_inputs = 1024  # TODO: modify for UNI (1024), ViT (768), ResNet50 (2048)
    n_hiddens = 512
    dropout = 0.2
    batch_size = 32
    learning_rate = 0.0001  ## 0.001
    n_outputs = len(all_target[0])
    max_epochs, patience = 300, 50

    ##================================================================================================
    ## model
    bias_init = torch.nn.Parameter(
        torch.Tensor(
            np.mean([sample[1].detach().cpu().numpy() for sample in train_set], axis=0)
        ).to(device)
    )

    model = MLP_regression(n_inputs, n_hiddens, n_outputs, dropout, bias_init)
    model.to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    ##================================================================================================
    print(" ")
    print(" --- fit --- ")
    start_time = time.time()

    # (
    #     model,
    #     train_loss,
    #     train_coef,
    #     train_slope,
    #     valid_loss,
    #     valid_coef,
    #     valid_slope,
    #     valid_labels,
    #     valid_preds,
    # ) 
    model_result = fit(model, optimizer, train_set, valid_set, max_epochs, patience, batch_size,results_path)
    print("\n\nfit -- completed -- time: {:.2f}".format(time.time() - start_time))
    return model_result
    ##================================================================================================

def result(model_results, results_path, genes_to_predict, test_set):
    # print(" ")
    print(" --- analyze_result --- ")
    start_time = time.time()
    (
        model,
        train_loss,
        train_coef,
        train_slope,
        valid_loss,
        valid_coef,
        valid_slope,
        valid_labels,
        valid_preds,
    ) = model_results

    analyze_result(
        results_path,
        genes_to_predict,
        model,
        train_loss,
        train_coef,
        train_slope,
        valid_loss,
        valid_coef,
        valid_slope,
        valid_labels,
        valid_preds,
        test_set,
    )

    print(f"analyze_result -- completed -- time: {(time.time() - start_time):.2f}s")
    ##================================================================================================

    print("--- completed ---")


parser = argparse.ArgumentParser(description='train gene expression')
parser.add_argument('--cancer_type', type=str,default='BRCA',choices=['BRCA','LUAD','STAD']) # Add others later
parser.add_argument('--curr_split', type=int, default=24)
parser.add_argument('--feature_extract', type=str, default= 'uni', choices=['uni','resnet'])
parser.add_argument('--base_input_path', type=str, default="/shares/sinha/lliu/projects/pre-cancer-image-omics",
                    help="base input path where all data is kept")
parser.add_argument('--output_path', type=str, default='/shares/sinha/sadeleye/ecPATH_Results')
parser.add_argument('--analyze_results', default=True, action='store_true')

BASE = ""

if __name__ == '__main__':
    args = parser.parse_args()
    BASE = args.base_input_path
    # BASE = os.path.join("/shares", "sinha", "lliu", "projects", "pre-cancer-image-omics")

    # 5 fold cross validation _ set up index
    random_seed = 0
    n_split = 5
    n_outers = n_split
    n_inners = n_split

    # all_cancer_types = os.listdir(os.path.join(BASE, "rawData", "slides"))
    # all_cancer_types = [item for item in all_cancer_types if not item.startswith(".")]

    cur_type = args.cancer_type 
    cur_split_selection = args.curr_split
    # sys.argv[1]  #'BRCA' #sys arg
    # cur_split_selection = int(sys.argv[2])  # sys arg

    sample_split_path = os.path.join(
        BASE,
        "rawData",
        "sample_split",
        f"sample_split_{n_outers}_{n_inners}_fold_{cur_type}.pkl",
    )

    # results_path = os.path.join(
    #     BASE,
    #     "predictResults",
    #     cur_type,
    #     f"{n_split}_folds_cur_{cur_split_selection}_resnet",
    #     "",  # TODO: modify for UNI (_uni0d), ResNet50
    # )
    results_path = os.path.join(args.output_path,
        "predictResults",
        cur_type,
        f"{n_split}_folds_cur_{cur_split_selection}_{args.feature_extract}",
        "",  # TODO: modify for UNI (_uni0d), ResNet50 (_resnet)
    )

    if not os.path.exists(results_path):
        print("output path created: ", results_path)
        os.makedirs(results_path)
    else:
        print("output path exists: ", results_path)

    print("output path: ", results_path)

    cur_results_len = len(os.listdir(results_path))

    print("cur_results_len-", cur_results_len)

    train_set, valid_set, test_set, genes_to_predict, all_target = dataPrep(cur_results_len, sample_split_path, results_path,args.feature_extract)

    model_results = train(all_target, train_set, valid_set)

    # if args.anayze_result:
    if args.analyze_results:
        result(model_results, results_path, genes_to_predict, test_set)
    