
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import random
import argparse
import scipy.stats
import pyreadr
import time
import os
from ecdna_label_model import *


#### Model description
# Build a model to predict ecDNA status from DeepPT predicted or true gene expression
# Take expression for genes and ecDNA status for samples
# Select top genes in training samples (according to univariate AUC)
# Use this genes as features for models, and calculate a weighted average across models



def data_stats(MLecDNA_paths, output_path='MLecdna_summary.csv'):
    # Initialize a list to store summary data
    summary_data = []
    
    for path in MLecDNA_paths:
        print(f"Now loading path: {path} .....")
        time_start = time.time()
        # Get the filename without extension as the DataFrame name
        df_name = os.path.basename(path).split('.')[0]
        
        # Read the R file
        MLecDNA_df_r = pyreadr.read_r(path)
        MLecDNA_df = MLecDNA_df_r[None]
        
        # Create a DataFrame with ecDNA status per sample
        sample_ecDNA_status_df = MLecDNA_df.groupby('sample').agg({
            'gene_class': lambda x: '1' if 'circular' in set(x) else '0'
        }).reset_index()
        
        # Count total samples
        total_samples = len(sample_ecDNA_status_df)
        
        # Count ecDNA+ and ecDNA- samples
        positive_count = sum(sample_ecDNA_status_df['gene_class'] == '1')
        negative_count = sum(sample_ecDNA_status_df['gene_class'] == '0')
        
        # Append the summary for this dataframe
        summary_data.append({
            'DataframeName': df_name,
            'Number_of_samples': total_samples,
            'Positive_ecdna_count': positive_count,
            'Negative_ecdna_count': negative_count
        })
        time_elapsed = time.time() - time_start
        print(f"Done, this took {time_elapsed} s'")
    
    # Convert summary data to a dataframe
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    summary_df.to_csv(output_path, index=False)
    
    print(f"Summary saved to {output_path}")
    return summary_df


def rank_normalize(array):
    # Get ranks (using 'rankdata' which ranks the data, ties are averaged)
    ranks = np.apply_along_axis(lambda x: scipy.stats.rankdata(x), axis=0, arr=array)
    
    # Normalize ranks to [0, 1]
    # Subtract 1 from ranks to start from 0, then divide by the max rank (n-1)
    normalized_ranks = (ranks - 1) / (array.shape[0] - 1)
    
    return normalized_ranks


def data_prep_Mlecdna(true_expression_df,predicted_expression_df,MLecDNA_df,cancer_type,exp_model, output_path,):
    # Setting 'ID' as index for easier reordering
    true_expression_df.set_index('sample_name', inplace=True)
    predicted_expression_df.set_index('slide_submitter_id', inplace=True)

    # Reindexing df2 to match the order of df1
    predicted_expression_df_reordered = predicted_expression_df.reindex(true_expression_df.index)

    # Resetting index if necessary
    predicted_expression_df_reordered.reset_index(inplace=True)
    predicted_expression_df = predicted_expression_df_reordered
    true_expression_df.reset_index(inplace=True)

    # Filter and aggregate gene ecdna prediction from MLecDna paper
    sample_ecDNA_status_df = MLecDNA_df.groupby('sample').agg({
            'gene_class': lambda x: '1' if 'circular' in set(x) else '0'
            }).reset_index()
    sample_ecDNA_status_df = sample_ecDNA_status_df.rename(columns={'gene_class': 'ecDNA_status'}) # rename to ecDna_status
    MLecDNA_samples = sample_ecDNA_status_df['sample'].tolist()

    # filter true_gene_expressions and predicted by samples predicted by MLecDna paper
    true_expression_df_filtered = true_expression_df[true_expression_df['sample_name'].str[:-1].isin(MLecDNA_samples)]
    pred_expression_df_filtered = predicted_expression_df[predicted_expression_df['sample_name'].str[:-1].isin(MLecDNA_samples)]

    # merge MLecDna predicitons with true gene epxressions:
    '''True gene expressions samples have sample_name: TCGA-A1-A0SM-01A, 
        while MLecDNA predictions have sample: TCGA-A1-A0SM-01. 
        We check if sample is sample name, and match the ecdna predictions as such. 
    '''
    sample_to_status = dict(zip(
        sample_ecDNA_status_df['sample'],
        sample_ecDNA_status_df['ecDNA_status']
    ))
    true_expression_df_filtered['sample_id'] = true_expression_df_filtered['sample_name'].str[:-1]  # Remove last character
    pred_expression_df_filtered['sample_id'] = pred_expression_df_filtered['sample_name'].str[:-1]  # Remove last character
    true_expression_df_filtered['ecDNA_status'] = true_expression_df_filtered['sample_id'].map(sample_to_status)
    pred_expression_df_filtered['ecDNA_status'] = pred_expression_df_filtered['sample_id'].map(sample_to_status)

    all_columns = true_expression_df_filtered.columns.tolist()
    all_columns.remove('sample_name')
    all_columns.remove('sample_id')
    all_columns.remove('ecDNA_status')
    new_order = ['sample_name', 'sample_id', 'ecDNA_status'] + all_columns[:]

    # Change columns to samplename ecdna_status sample_id and genes
    merged_df_true_filtered = true_expression_df_filtered[new_order]
    merged_df_true_filtered = merged_df_true_filtered.reset_index(drop=True)
    
    merged_df_pred_filtered = pred_expression_df_filtered[new_order]
    merged_df_pred_filtered = merged_df_true_filtered.reset_index(drop=True)

    merged_df_pred_filtered = merged_df_pred_filtered.rename(columns={'sample_id': 'patient_id'}) 
    merged_df_true_filtered = merged_df_true_filtered.rename(columns={'sample_id': 'patient_id'}) 


    # Save prediction files 
    # merged_df_pred_filtered.to_csv(output_path+ f"/TCGA_{cancer_type}_new_ecDNA_samples_for_prediction_from_DeepPT_exp_{exp_model}.csv", index=False)

    # Data preparation
    X_true = merged_df_true_filtered.iloc[:, 3:].values  # all rows, all columns except the last one
    y = merged_df_true_filtered.iloc[:, 2].values  # all rows, last column
    patients = merged_df_true_filtered.iloc[:, 1].values
    X_pred = merged_df_pred_filtered.iloc[:, 3:].values  # all rows, all columns except the last one

    X_rank_normalized_pred = rank_normalize(X_pred)
    X_rank_normalized_true = rank_normalize(X_true)

    return ( X_rank_normalized_true, X_rank_normalized_pred, patients, y), merged_df_true_filtered,  merged_df_pred_filtered, all_columns

def data_prep_DeepPT(cancer_type,corr_threshold,q_value_threshold, genes_info_file,true_expression_df,
                     predicted_expression_df,ecDNA_df,output_path,exp_model):
    
    # Setting 'ID' as index for easier reordering
    true_expression_df.set_index('sample_name', inplace=True)
    predicted_expression_df.set_index('slide_submitter_id', inplace=True)

    # Reindexing df2 to match the order of df1
    predicted_expression_df_reordered = predicted_expression_df.reindex(true_expression_df.index)

    # Resetting index if necessary
    predicted_expression_df_reordered.reset_index(inplace=True)
    predicted_expression_df = predicted_expression_df_reordered
    true_expression_df.reset_index(inplace=True)


    # ecDna status from DeepPt
    project_id = f"TCGA-{cancer_type}"
    ecDNA_cancer_df = ecDNA_df[ecDNA_df['cancer_type'] == project_id]

    # Get the genes that can be predicted by the DeepPT model (q < 0.05 and corr > 0.4)
    gene_info = pd.read_csv(genes_info_file, delimiter="\t")
    # Filter genes with p_adj < 0.05
    genes_of_interest_df = gene_info[gene_info['Pearson_padj'] < q_value_threshold]
    # Further filter genes with coef > 0.4
    genes_of_interest_df = genes_of_interest_df[genes_of_interest_df['Pearson_corr'] > corr_threshold]
    genes_of_interest = list(genes_of_interest_df['Gene_ENSID'])

    columns = ['sample_name'] + genes_of_interest
    #Keep only sample_names and the DeepPT predictable genes in expression file
    select_expression_df_pred = predicted_expression_df[columns]
    select_expression_df_true = true_expression_df[columns]

     # keep ecDNA status and sample names and merge with expression file
    columns_to_keep = ['sample', 'patient_id', 'ecDNA_status']
    # Create a new DataFrame with only the selected columns
    ecDNA_df_selected = ecDNA_cancer_df[columns_to_keep]

    # Drop duplicate rows based on the selected columns
    ecDNA_df = ecDNA_df_selected.drop_duplicates()


    columns_in_merged = ['sample_name','patient_id','ecDNA_status'] + genes_of_interest
    merged_df_true = pd.merge(ecDNA_df, select_expression_df_true, right_on='sample_name', left_on="sample", how='inner')
    merged_df_pred = pd.merge(ecDNA_df, select_expression_df_pred, right_on='sample_name', left_on="sample", how='inner')

    merged_df_true = merged_df_true[columns_in_merged]
    merged_df_pred = merged_df_pred[columns_in_merged]

    output_ecDNA_sample_df = merged_df_true[['sample_name','patient_id','ecDNA_status']]
    # output_ecDNA_sample_df.to_csv(output_path+ f"/TCGA_{cancer_type}_new_ecDNA_samples_for_prediction_from_DeepPT_exp_{exp_model}.csv", index=False)

    # Data preparation
    X_true = merged_df_true.iloc[:, 3:].values 
    X_pred = merged_df_pred.iloc[:, 3:].values  
    y = merged_df_true.iloc[:, 2].values  # all rows, last column
    patients = merged_df_true.iloc[:, 1].values

    X_rank_normalized_pred = rank_normalize(X_pred)
    X_rank_normalized_true = rank_normalize(X_true)

    
    return ( X_rank_normalized_true, X_rank_normalized_pred, patients, y), merged_df_true,  merged_df_pred, genes_of_interest


def train(model_results, repeats, k_fold_splits, genes_of_interest,  expression_type, ml_method, number_top_features,
          output_model_file_fn, output_gene_feature_file_fn, output_gene_feature_csv_file_fn,output_auc_score_per_fold_file_fn):
    ( 
        X_rank_normalized_true, 
        X_rank_normalized_pred, 
        patients,
        y, 
     ) = model_results
    ######### Build the weighted or average ensembl model running repeated stratified k-fold cross validation

    auc_scores_per_fold = []
    model_pred_dfs = []
    auc_scores = []
    include_low_AUC = True #this means that when extracting top gene features the script uses genes with both high and low univariate AUC values
    models = []
    gene_features_for_models = []
    gene_feature_AUCs_for_models = []
    i = 0
    for repeat in range(repeats):

        # Set up group stratified k-fold
        cv_outer = StratifiedKFold(n_splits=k_fold_splits, random_state=repeat, shuffle=True)
        # Set up the inner cross-validation strategy for hyperparameter tuning
        cv_inner = StratifiedKFold(n_splits=k_fold_splits, random_state=repeat, shuffle=True)
        
        if expression_type == "true":
            x_data_to_split = X_rank_normalized_true
        if expression_type == "predicted":
            x_data_to_split = X_rank_normalized_pred
        
        #for train_idx, test_idx in cv_outer.split(x_data_to_split, y, groups = patients): #split the data into train and test but group by the patient IDs
        
        ###### If grouping by patient ID use the line below instead
        for train_idx, test_idx in cv_outer.split(x_data_to_split, y, groups = patients): #split the data into train and test
            i += 1
            print(f"training iteration: {i}")
            
            if expression_type == "predicted":
                X_train_pred_ex, X_test_pred_ex = X_rank_normalized_pred[train_idx], X_rank_normalized_pred[test_idx]
                X_train_true_ex = X_rank_normalized_true[train_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                patient_IDs_training = patients[train_idx]
                patient_IDs_test = patients[test_idx]
        
                # Assume custom feature selection is properly defined
                selected_features_indices, AUC_for_select_features, select_feature_names, gene_AUCs = select_features_based_on_predicted_training_data(X_train_pred_ex, X_train_true_ex, y_train, include_low_AUC, number_top_features,genes_of_interest)
                X_train_selected = X_train_pred_ex[:, selected_features_indices]
                X_test_selected = X_test_pred_ex[:, selected_features_indices]

        
            if expression_type == "true":
                X_train_true_ex, X_test_true_ex = X_rank_normalized_true[train_idx], X_rank_normalized_true[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                patient_IDs_training = patients[train_idx]
                patient_IDs_test = patients[test_idx]
      
        
                # Assume custom feature selection is properly defined
                selected_features_indices, AUC_for_select_features, select_feature_names, gene_AUCs = select_features_based_on_true_training_data(X_train_true_ex, y_train, include_low_AUC, number_top_features,genes_of_interest)
                X_train_selected = X_train_true_ex[:, selected_features_indices]
                X_test_selected = X_test_true_ex[:, selected_features_indices]

            
            inner_splits = list(cv_inner.split(X_train_selected, y_train))

            # Fit models on the train set
            print("--- Running model")
            if ml_method == "LR":
                predictions, best_model, AUC, pred_df = Logistic_Regression_model(X_train_selected,X_test_selected, 
                                                                                        y_train, y_test, test_idx, i, inner_splits, patient_IDs_training)
            elif ml_method == "RF":
                predictions, best_model, AUC, pred_df = Random_Forest_model(X_train_selected, X_test_selected, 
                                                                                    y_train, y_test, test_idx, i, inner_splits, patient_IDs_training)
            elif ml_method == "GB":
                predictions, best_model, AUC, pred_df = Gradient_Boosting_model(X_train_selected,X_test_selected, 
                                                                                        y_train, y_test, test_idx, i, inner_splits, patient_IDs_training)
            elif ml_method == "SVM":
                predictions, best_model, AUC, pred_df = SVM_model(X_train_selected, X_test_selected, 
                                                                            y_train, y_test, test_idx, i, inner_splits, patient_IDs_training)
            

            print(f"--- {ml_method} auc = {round(AUC,3)}")
            auc_scores.append(AUC)


            #Create a dataframe with all the weighted predictions from all models
            pred_data = {'repeat': np.repeat(i, len(test_idx)), 'sample_idx': test_idx, 'Prediction': predictions}
            new_pred_df = pd.DataFrame(pred_data)
            model_pred_dfs.append(new_pred_df)
 

            models.append(best_model)
            gene_features_for_models.append(select_feature_names)
            gene_feature_AUCs_for_models.append(gene_AUCs)
            print(select_feature_names[0:5])
            print(gene_AUCs[0:5])
            
            print('Average AUC so far: %.3f (±%.3f)' % (np.mean(auc_scores), np.std(auc_scores)))


    # Display the average AUC over all rounds
    print('Final Average AUC: %.3f (±%.3f)' % (np.mean(auc_scores), np.std(auc_scores)))

    # save model
    joblib.dump(models, output_model_file_fn)
    select_feature_names_df = pd.DataFrame(gene_features_for_models)
    select_feature_names_df.to_pickle(output_gene_feature_file_fn)


    auc_scores_per_fold_df = pd.DataFrame(columns=['Fold', 'AUC_Score', 'AUC_STD'])
    for fold_idx, auc_curr in enumerate(auc_scores):
        auc_scores_per_fold_df.loc[fold_idx] = [f'Fold {fold_idx+1}', auc_curr, 0]
    auc_scores_per_fold_df.loc[len(auc_scores_per_fold_df)] = ['Total Average', np.mean(auc_scores), np.std(auc_scores)]
    auc_scores_per_fold_df.to_csv(output_auc_score_per_fold_file_fn)



    gene_feature_list = [item for sublist in gene_features_for_models for item in sublist]
    auc_feature_list = [item for sublist in gene_feature_AUCs_for_models for item in sublist]
    select_feature_and_AUC_names_df = pd.DataFrame({'Gene_name': gene_feature_list, 'AUC': auc_feature_list})
    select_feature_and_AUC_names_df.to_csv(output_gene_feature_csv_file_fn)

    return auc_scores,model_pred_dfs


def predict_and_results(auc_scores, model_pred_dfs, prediction_result_fn, merged_df_pred):

    auc_output = round(np.mean(auc_scores),3)
    print(auc_output)

    combined_pred_df = pd.concat(model_pred_dfs, ignore_index=True)
    # Calculate the mean of 'Prediction' for each 'sample_idx'
    average_predictions = pd.DataFrame(combined_pred_df.groupby('sample_idx')['Prediction'].mean())

    prediction_result_df = pd.concat([merged_df_pred[['sample_name','patient_id','ecDNA_status']], average_predictions[['Prediction']]], axis=1)
    
    prediction_result_df.to_csv(prediction_result_fn, index=False)


    print("job completed")




parser = argparse.ArgumentParser(description='train gene status')
parser.add_argument('--cancer_type', type=str,default='BRCA',choices=['BRCA','LUAD','STAD','HNSC','LGG','CESC','LUSC','ESCA','GBM']) 
parser.add_argument('--ml_methods', type=str, default='LR', choices=['LR', 'GB', 'SVM', 'RF'], 
                    help= "logistic regression, gradient boosting, SVM, Random Forest")
parser.add_argument('--expression_type', type=str, default='true',choices=['true','predicted']) 
parser.add_argument('--k_fold_splits', type=int, default=5)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--feature_extract', type=str, default= 'uni', choices=['uni','resnet'])
parser.add_argument('--data_type', type=str, default= 'MLecdna', choices=['DeepPt','MLecdna'])

parser.add_argument('--base_input_path', type=str, default="/shares/sinha/mchoudhury/projects/pre-cancer-image-omics/TCGA_all_cancer_DeepPT",
                    help="base input path where all data is kept")
parser.add_argument('--output_path', type=str, default='/shares/sinha/sadeleye/ecPATH_Results/ecdna_LabelResult')

parser.add_argument('--corr_threshold', type=int, default=0.4) #DeepPT correlation threshold
parser.add_argument('--q_value_threshold', type=int, default=0.05) #DeepPT p adj threshold
parser.add_argument('--top_gene_features', type=int, default=150) 
args = parser.parse_args()


if __name__ == '__main__':
    print("Starting....")
    
    cancer_type = args.cancer_type
    ml_method = args.ml_methods  # LR, GB, SVM, RF (logistic regression, gradient boosting, SVM, RF)
    expression_type = args.expression_type  # "true" or "predicted"
    k_fold_splits = args.k_fold_splits  # usually 5
    repeats = args.epochs 
    exp_model = args.feature_extract
    data_type = args.data_type

    corr_threshold = args.corr_threshold  # DeepPT correlation threshold
    q_value_threshold = args.q_value_threshold  # DeepPT p adj threshold
    number_top_features = args.top_gene_features

    random.seed(42)

    # Set random seed for numpy
    np.random.seed(42)
    
    # cancer_types = ['BRCA','LUAD','STAD','HNSC','LGG','CESC','LUSC','ESCA','GBM']
    cancer_types = [cancer_type]
    for cancer_type in cancer_types:
        # output predictions file:
        if expression_type == "true":
            prediction_result_fn = args.output_path + f"/TCGA_{cancer_type}_{ml_method}_method_mean_ecDNA_{data_type}_predictions_nested_{k_fold_splits}_fold_{repeats}_repeat_on_{expression_type}_ex_DeepPT_.csv"
        if expression_type == "predicted":
            prediction_result_fn = args.output_path + f"/TCGA_{cancer_type}_{ml_method}_method_mean_ecDNA_{data_type}_predictions_nested_{k_fold_splits}_fold_{repeats}_repeat_on_{expression_type}_ex_DeepPT_{exp_model}.csv"
            
        ## output models and genes for models files:
        output_model_file_fn = args.output_path + f"/TCGA_ecDNA_models/TCGA_{cancer_type}_{k_fold_splits}_split_{repeats}_repeat_{expression_type}_expression_{ml_method}_models_{exp_model}_{data_type}.pkl"
        output_gene_feature_file_fn = args.output_path + f"/TCGA_ecDNA_models/TCGA_{cancer_type}_{k_fold_splits}_split_{repeats}_repeat_{expression_type}_expression_{ml_method}_models_gene_features_{exp_model}_{data_type}.pkl"
        output_gene_feature_csv_file_fn = args.output_path + f"/TCGA_ecDNA_models/TCGA_{cancer_type}_{k_fold_splits}_split_{repeats}_repeat_{expression_type}_expression_{ml_method}_models_gene_features_{exp_model}_{data_type}.csv"
        output_auc_score_per_fold_file_fn = args.output_path + f"/TCGA_ecDNA_models/TCGA_{cancer_type}_{k_fold_splits}_split_{repeats}_repeat_{expression_type}_expression_{ml_method}_auc_scores_{exp_model}_{data_type}.csv"
        
        if not os.path.exists(args.output_path+"/TCGA_ecDNA_models/"):
            print("output path created: ", args.output_path+"/TCGA_ecDNA_models/")
            os.makedirs(args.output_path+"/TCGA_ecDNA_models/")
    
        # Read expression file and ecDNA file
        # columns: Sample_name Gene expression1 ....
        true_expression_df = pd.read_csv(args.base_input_path 
                    + f"/data/TCGA_{cancer_type}.DeepPT_normalized_reformatted_true_expression_per_sample_{exp_model}.txt",
                    delimiter='\t')
        predicted_expression_df = pd.read_csv(args.base_input_path 
                    + f"/data/TCGA_{cancer_type}.DeepPT_reformatted_avg_predicted_expression_per_sample_{exp_model}.txt",
                    delimiter='\t')  # reorder the samples here
        
        # All tumor TCGA info. from DeepPt
        # columns: sample, patient_id, cancer_type.., ecDNA_status
        ecDNA_df = pd.read_csv(args.base_input_path + '/../TCGA_eCDNA/data/TCGA_tumor_samples_all_cancer_type_ecDNA_and_other_variant_status.csv')

        # All tumor TCGA info from MLecdna
        MLecDNA_df_r = pyreadr.read_r(f'/shares/sinha/sadeleye/ecPATH/Data/Training_Data/tcga_snp_array_gcap_result2/TCGA_SNP_{cancer_type}_prediction_result.rds')
        MLecDNA_df = MLecDNA_df_r[None] 

        # Gene info file
        genes_info_file = args.base_input_path + f"/results/TCGA_{cancer_type}.DeepPT_pred_coef_pvalue_per_gene_{exp_model}.txt"

        print("Loading data...")        
        if data_type == "DeepPt":
            model_results, merged_df_true, merged_df_pred, genes_of_interest =  data_prep_DeepPT(cancer_type,corr_threshold,q_value_threshold, genes_info_file,true_expression_df,
                        predicted_expression_df,ecDNA_df,args.output_path,exp_model)
        else:
            model_results, merged_df_true, merged_df_pred, genes_of_interest =  data_prep_Mlecdna(true_expression_df,predicted_expression_df,MLecDNA_df,cancer_type,exp_model,args.output_path)
        
        
        print("Data Loaded \n Now Starting to Train Model... ")
        auc_scores, model_pred_dfs = train( model_results, repeats, k_fold_splits, genes_of_interest,
                                            expression_type, ml_method, number_top_features,
                                            output_model_file_fn, 
                                            output_gene_feature_file_fn,
                                            output_gene_feature_csv_file_fn,
                                            output_auc_score_per_fold_file_fn
                                        )
        print("Training Finished. Saving Results..")
        predict_and_results(auc_scores, model_pred_dfs, prediction_result_fn, merged_df_true)
        print(f"Results saved at {args.output_path}")
