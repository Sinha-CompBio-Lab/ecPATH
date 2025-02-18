#!/usr/bin/env python
# coding: utf-8

# In[28]:


#### Model description
# Build a model to predict ecDNA status from DeepPT predicted gene expression
# Take predicted expression for genes that can be reliably predicted by DeepPT
# Take ecDNA status for samples
# Select top genes in training samples (according to univariate AUC)
# Use this genes as features for models, and calculate a weighted average across models


# In[122]:

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import matplotlib.pyplot as plt
import warnings
import sys
import random
import argparse
import scipy.stats
from sklearn.model_selection import StratifiedGroupKFold


def rank_normalize(array):
    # Get ranks (using 'rankdata' which ranks the data, ties are averaged)
    ranks = np.apply_along_axis(lambda x: scipy.stats.rankdata(x), axis=0, arr=array)
    
    # Normalize ranks to [0, 1]
    # Subtract 1 from ranks to start from 0, then divide by the max rank (n-1)
    normalized_ranks = (ranks - 1) / (array.shape[0] - 1)
    
    return normalized_ranks


#  In[31]:
def data_prep(cancer_type,exp_model,corr_threshold,q_value_threshold,output_path, genes_info_file,true_expression_df,predicted_expression_df,ecDNA_df):
    # Setting 'ID' as index for easier reordering
    true_expression_df.set_index('sample_name', inplace=True)
    predicted_expression_df.set_index('slide_submitter_id', inplace=True)

    # Reindexing df2 to match the order of df1
    predicted_expression_df_reordered = predicted_expression_df.reindex(true_expression_df.index)
    true_expression_df_reordered = true_expression_df.reindex(true_expression_df.index)

    # Resetting index if necessary
    predicted_expression_df_reordered.reset_index(inplace=True)
    true_expression_df_reordered.reset_index(inplace=True)

    predicted_expression_df = predicted_expression_df_reordered
    true_expression_df = true_expression_df_reordered

    project_id = f"TCGA-{cancer_type}"
    ecDNA_cancer_df = ecDNA_df[ecDNA_df['cancer_type'] == project_id]

    print(true_expression_df.head())
    print(predicted_expression_df.head())
    print(ecDNA_cancer_df.head())


    # In[32]:


    # Get the genes that can be predicted by the DeepPT model (q < 0.05 and corr > 0.4)
    # Read the CSV file
    gene_info = pd.read_csv(genes_info_file, delimiter="\t")
    print(gene_info)

    # Filter genes with p_adj < 0.05
    genes_of_interest_df = gene_info[gene_info['Pearson_padj'] < q_value_threshold]

    # Further filter genes with coef > 0.4
    genes_of_interest_df = genes_of_interest_df[genes_of_interest_df['Pearson_corr'] > corr_threshold]
    genes_of_interest = list(genes_of_interest_df['Gene_ENSID'])

    # Print the resulting DataFrame
    print(genes_of_interest[1:5])

    columns = ['sample_name'] + genes_of_interest
    #Keep only sample_names and the DeepPT predictable genes in expression file
    select_expression_df_pred = predicted_expression_df[columns]
    select_expression_df_true = true_expression_df[columns]

    print(select_expression_df_pred.head())
    print(select_expression_df_true.head())


    # In[33]:


    # keep ecDNA status and sample names and merge with expression file
    columns_to_keep = ['sample', 'patient_id', 'ecDNA_status']

    # Create a new DataFrame with only the selected columns
    ecDNA_df_selected = ecDNA_cancer_df[columns_to_keep]


    # Drop duplicate rows based on the selected columns
    ecDNA_df = ecDNA_df_selected.drop_duplicates()

    # Print the resulting DataFrame
    #print(ecDNA_df)

    merged_df_pred = pd.merge(ecDNA_df, select_expression_df_pred, right_on='sample_name', left_on="sample", how='inner')
    merged_df_true = pd.merge(ecDNA_df, select_expression_df_true, right_on='sample_name', left_on="sample", how='inner')
    columns_in_merged = ['sample_name','patient_id','ecDNA_status'] + genes_of_interest
    merged_df_pred = merged_df_pred[columns_in_merged]
    merged_df_true = merged_df_true[columns_in_merged]
    print(merged_df_pred.head())
    print(merged_df_true.head())

    output_ecDNA_sample_df = merged_df_pred[['sample_name','patient_id','ecDNA_status']]
    ecDNA_status = pd.Series(merged_df_pred['ecDNA_status'])
    print(ecDNA_status.value_counts())
    output_ecDNA_sample_df.to_csv(output_path+ f"/TCGA_{cancer_type}_new_ecDNA_samples_for_prediction_from_DeepPT_exp_{exp_model}.csv", index=False)


    # In[34]:


    # Data preparation
    X_pred = merged_df_pred.iloc[:, 3:].values  # all rows, all columns except the last one
    X_true = merged_df_true.iloc[:, 3:].values  # all rows, all columns except the last one
    y = merged_df_pred.iloc[:, 2].values  # all rows, last column
    patients = merged_df_pred.iloc[:, 1].values
    print(X_pred)
    print(X_true)
    print(y)
    print(patients)


    # In[35]:

    X_rank_normalized_pred = rank_normalize(X_pred)
    X_rank_normalized_true = rank_normalize(X_true)

    print(X_rank_normalized_pred)
    print(X_rank_normalized_true)
    print(y)
    return ( X_rank_normalized_true, X_rank_normalized_pred, patients, y), merged_df_pred, genes_of_interest



#Define the select_features_based_on_training_data function for predicted expression of genes
#This function takes average univariate AUC for true and predicted expression and selects the top and bottom as features

def select_features_based_on_predicted_training_data(X_train_predicted_ex, X_train_true_ex, y_train, include_low_AUC, number_top_features,genes_of_interest):
    #number_top_features = 150
    """
    Select the top features based on univariate AUC scores.
    
    Parameters:
    - X_train: Training features (numpy array)
    - y_train: Training target variable (numpy array)
    
    Returns:
    - List of indices for the top 100 features with the highest AUC scores.
    """
    auc_scores = []
    print("---select features")
    # Iterate over each feature in X_train
    for i in range(X_train_predicted_ex.shape[1]):
        # Try to compute AUC, if the feature is constant or nearly constant, it might fail
        try:
            # Score the feature
            score_pred_ex = roc_auc_score(y_train, X_train_predicted_ex[:, i])
            score_true_ex = roc_auc_score(y_train, X_train_true_ex[:, i])
            score = (score_pred_ex + score_true_ex)/2
            auc_scores.append(score)
        except ValueError:
            # If there is an error (e.g., only one class present in y_true), append a low score
            auc_scores.append(None)
    
    # Convert scores to a numpy array
    true_auc_scores = np.array(auc_scores)
    if include_low_AUC == True:
        auc_scores = np.array([1 - x if x is not None and x < 0.5 else x for x in true_auc_scores])
        #auc_scores = [1 - x if x < 0.5 else x for x in auc_scores]
        #auc_scores = np.array(auc_scores)

    #print(auc_scores)
    
    # Get the indices of the top 100 features
    top_features_indices = np.argsort(auc_scores)[::-1][:number_top_features]
    #print("select features AUC scores")
    #print(auc_scores[top_features_indices])
    avg_uni_auc = np.mean(auc_scores[top_features_indices])
    #print(top_features_indices)
    top_feature_genes = [genes_of_interest[k] for k in top_features_indices]
    top_gene_aucs = true_auc_scores[top_features_indices]

    ### For testing
    print(f"--- Univariate avg AUCs: {avg_uni_auc}")
    #print(auc_scores)
    #print(np.argsort(auc_scores)[::-1])
    #print(top_features_indices)
    #print(auc_scores[top_features_indices])
    #print(avg_uni_auc)
    
    return top_features_indices, avg_uni_auc, top_feature_genes, top_gene_aucs


# In[146]:


#Define the select_features_based_on_training_data function for true expression of genes
#This function takes average univariate AUC for true expression and selects the top and bottom as features

def select_features_based_on_true_training_data(X_train_true_ex, y_train, include_low_AUC, number_top_features,genes_of_interest):
    #number_top_features = 150
    """
    Select the top features based on univariate AUC scores.
    
    Parameters:
    - X_train: Training features (numpy array)
    - y_train: Training target variable (numpy array)
    
    Returns:
    - List of indices for the top 100 features with the highest AUC scores.
    """
    auc_scores = []
    print("---select features")
    #print(X_train_true_ex)
    #print(y_train)
    # Iterate over each feature in X_train
    for i in range(X_train_true_ex.shape[1]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Try to compute AUC, if the feature is constant or nearly constant, it might fail
            try:
                score = roc_auc_score(y_train, X_train_true_ex[:, i])
                auc_scores.append(score)
            except Exception:
                #print("failed to compute")
                # If there is an error (e.g., only one class present in y_true), append a low score
                auc_scores.append(None)
    
    # Convert scores to a numpy array
    true_auc_scores = np.array(auc_scores)
    if include_low_AUC == True:
        auc_scores = np.array([1 - x if x is not None and x < 0.5 else x for x in true_auc_scores])
        #auc_scores = [1 - x if x < 0.5 else x for x in auc_scores]
        #auc_scores = np.array(auc_scores)

    #print(auc_scores)
    
    # Get the indices of the top 100 features
    top_features_indices = np.argsort(auc_scores)[::-1][:number_top_features]
    #print("select features AUC scores")
    #print(auc_scores[top_features_indices])
    avg_uni_auc = np.mean(auc_scores[top_features_indices])
    #print(top_features_indices)
    top_feature_genes = [genes_of_interest[k] for k in top_features_indices]
    top_gene_aucs = true_auc_scores[top_features_indices]

    ### For testing
    print(f"--- Univariate avg AUCs: {avg_uni_auc}")
    #print(auc_scores)
    #print(np.argsort(auc_scores)[::-1])
    #print(top_features_indices)
    #print(auc_scores[top_features_indices])
    #print(avg_uni_auc)
    print("---done selecting")
    return top_features_indices, avg_uni_auc, top_feature_genes, top_gene_aucs


# In[133]:


#Build SVM model

def SVM_model(X_train_selected, X_test_selected, y_train, y_test, test_idx, i, inner_splits, patients):
    # Define the model
    print("--- SVM...")
    svc = SVC(probability=True, random_state = 1)
    #inner_splits = list(cv_inner.split(X_train_selected, y_train, groups=patients))
    #inner_splits = list(cv_inner.split(X_train_selected, y_train))
    # Define parameters for GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [0.01, 0.1, 1],
        'kernel': ['rbf', 'poly']
    }

    # Setup GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, n_jobs=-1, cv=inner_splits, scoring=make_scorer(roc_auc_score))

    # Fit on the train set
    result = grid_search.fit(X_train_selected, y_train)

    # Evaluate on the test set
    best_model = result.best_estimator_
    y_pred = best_model.predict_proba(X_test_selected)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    #print('SVM AUC: %.3f' % auc)
    #print(f"best model: {best_model}")
    
    pred_data = {'Model': ['SVM'] * len(y_pred),'iteration': [i] * len(y_pred),'sample_idx': test_idx, 'Prediction': y_pred, 'AUC': [auc] * len(y_pred)}
    pred_df = pd.DataFrame(pred_data)
    #print(auc)

    
    return y_pred, best_model, auc, pred_df


# In[134]:


#Build Random Forest model

def Random_Forest_model(X_train_selected, X_test_selected, y_train, y_test, test_idx, i, inner_splits, patients):
    print("--- Random forest...")
    grid_search_iterations = 50
    #inner_splits = list(cv_inner.split(X_train_selected, y_train, groups=patients))
    #inner_splits = list(cv_inner.split(X_train_selected, y_train))
    
    # Define the model
    model = RandomForestClassifier(random_state=1)

    # Define parameters for GridSearchCV
    param_grid = {
        'n_estimators': np.arange(10, 100, 10),
        'max_features': ['sqrt','log2'],
        'min_samples_split': np.arange(2, 8, 2),
        'min_samples_leaf': np.arange(1, 5, 1),
        'max_depth': np.arange(2, 11, 2)
    }

    # Configure random grid search
    search = RandomizedSearchCV(model, param_grid, n_iter=grid_search_iterations, scoring='roc_auc', cv=inner_splits, refit=True, n_jobs=-1)

    # Execute search
    result = search.fit(X_train_selected, y_train)

    # Get the best model
    best_model = result.best_estimator_

    # Evaluate the best model on the holdout set using AUC
    y_pred = best_model.predict_proba(X_test_selected)[:, 1]  # get probabilities for the positive class
    auc = roc_auc_score(y_test, y_pred)

    pred_data = {'Model': ['RF'] * len(y_pred),'iteration': [i] * len(y_pred),'sample_idx': test_idx, 'Prediction': y_pred, 'AUC': [auc] * len(y_pred)}
    pred_df = pd.DataFrame(pred_data)
    #print(auc)
    
    return y_pred, best_model, auc, pred_df


# In[135]:


#Build Gradient Boosting model

def Gradient_Boosting_model(X_train_selected, X_test_selected, y_train, y_test, test_idx, i, inner_splits, patients):
    print("--- Gradient Boosting...")

    #inner_splits = list(cv_inner.split(X_train_selected, y_train, groups=patients))
    #inner_splits = list(cv_inner.split(X_train_selected, y_train))
    
    # Define the model
    gbm = GradientBoostingClassifier(random_state=42)

    # Define parameters for GridSearchCV
    param_grid = {
        'n_estimators': [10, 50, 100],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'subsample': [0.5, 0.6, 0.7, 0.8]  # Subsample ratio of the training set
    }

    # Configure GridSearchCV
    grid_search = GridSearchCV(estimator=gbm, param_grid=param_grid, scoring='roc_auc', cv=inner_splits, refit=True)


    # Fit the grid search to the training data
    result = grid_search.fit(X_train_selected, y_train)

    # Evaluate the best model on the test set
    best_model = result.best_estimator_
    y_pred = best_model.predict_proba(X_test_selected)[:, 1]  # Get probability estimates for the positive class
    auc = roc_auc_score(y_test, y_pred)

    pred_data = {'Model': ['GB'] * len(y_pred),'iteration': [i] * len(y_pred),'sample_idx': test_idx, 'Prediction': y_pred, 'AUC': [auc] * len(y_pred)}
    pred_df = pd.DataFrame(pred_data)
    #print(auc)
    
    return y_pred, best_model, auc, pred_df


# In[147]:


#Build Logistic Regression model

def Logistic_Regression_model(X_train_selected, X_test_selected, y_train, y_test, test_idx, i, inner_splits, patients):
    print("--- Logistic Regression...")

    #inner_splits = list(cv_inner.split(X_train_selected, y_train, groups=patients))
    #inner_splits = list(cv_inner.split(X_train_selected, y_train))
    
    # Parameter grid for Logistic Regression
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],  # Example regularization strengths
        'solver': ['liblinear']  # solvers appropriate for small to medium datasets and binary classification
    }

    # Define the model with GridSearchCV
    model = GridSearchCV(
        LogisticRegression(max_iter=10000, random_state=42),
        param_grid,
        scoring='roc_auc',
        cv=inner_splits,  # using the provided inner cross-validation strategy
        refit=True  # Refits the best model to the entire set of provided samples
    )

    # Fit the model with the selected features of the training data
    #print("is it here?")
    #print(X_train_selected)
    #print(y_train)
    model.fit(X_train_selected, y_train)

    # Retrieve the best model from GridSearchCV
    best_model = model.best_estimator_

    # Make predictions on the test set using the best model
    y_pred = best_model.predict_proba(X_test_selected)[:, 1]
    auc = roc_auc_score(y_test, y_pred)

    # Prepare the prediction data for output
    pred_data = {
        'Model': ['LR'] * len(y_pred),
        'iteration': [i] * len(y_pred),
        'sample_idx': test_idx,
        'Prediction': y_pred,
        'AUC': [auc] * len(y_pred)
    }
    pred_df = pd.DataFrame(pred_data)
    #print(auc)
    #print("done with LR")
    return y_pred, best_model, auc, pred_df


# In[173]:
def train(model_results, repeats, k_fold_splits, genes_of_interest,  expression_type, ml_method, number_top_features,
          output_model_file_fn, output_gene_feature_file_fn, output_gene_feature_csv_file_fn):
    ( 
        X_rank_normalized_true, 
        X_rank_normalized_pred, 
        patients,
        y, 
     ) = model_results
    ######### Build the weighted or average ensembl model running repeated stratified k-fold cross validation

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

        
        ##### if grouping by patient use this code, which is causing some issues with the test train split right now for some reason:
        # Set up group stratified k-fold
        #cv_outer = StratifiedGroupKFold(n_splits=k_fold_splits, random_state=repeat, shuffle=True)
        # Set up the inner cross-validation strategy for hyperparameter tuning
        #cv_inner = StratifiedGroupKFold(n_splits=k_fold_splits, random_state=repeat, shuffle=True)
        
        
        #all_model_df = []
        
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
                #print(X_train_true_ex)
                #print(y_train)
                #print(X_test_true_ex)
                #print(y_test)
        
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


    joblib.dump(models, output_model_file_fn)
    select_feature_names_df = pd.DataFrame(gene_features_for_models)
    select_feature_names_df.to_pickle(output_gene_feature_file_fn)
    #select_feature_names_df.to_csv(output_gene_feature_csv_file_fn)



    #print(select_feature_names_df)
    #select_feature_names_df.to_csv(output_gene_feature_csv_file_fn)
    gene_feature_list = [item for sublist in gene_features_for_models for item in sublist]
    auc_feature_list = [item for sublist in gene_feature_AUCs_for_models for item in sublist]
    select_feature_and_AUC_names_df = pd.DataFrame({'Gene_name': gene_feature_list, 'AUC': auc_feature_list})
    print(select_feature_and_AUC_names_df)
    select_feature_and_AUC_names_df.to_csv(output_gene_feature_csv_file_fn)

    return auc_scores,model_pred_dfs



# In[174]:
def predict_and_results(auc_scores, model_pred_dfs, prediction_result_fn, merged_df_pred):

    auc_output = round(np.mean(auc_scores),3)
    print(auc_output)


    combined_pred_df = pd.concat(model_pred_dfs, ignore_index=True)
    #print(n_estimators)
    # Display the concatenated DataFrame of predictions
    #print(combined_pred_df)
    # Calculate the mean of 'Prediction' for each 'sample_idx'
    average_predictions = pd.DataFrame(combined_pred_df.groupby('sample_idx')['Prediction'].mean())

    # Result
    #print(average_predictions)

    prediction_result_df = pd.concat([merged_df_pred[['sample_name','patient_id','ecDNA_status']], average_predictions[['Prediction']]], axis=1)
    prediction_result_df.to_csv(prediction_result_fn, index=False)
    #print(prediction_result_df)


    print("job completed")



# In[30]:
parser = argparse.ArgumentParser(description='train gene expression')
parser.add_argument('--cancer_type', type=str,default='BRCA',choices=['BRCA','LUAD','STAD']) 
parser.add_argument('--ml_methods', type=str, default='LR', choices=['LR', 'GB', 'SVM', 'RF'], 
                    help= "logistic regression, gradient boosting, SVM, Random Forest")
parser.add_argument('--expression_type', type=str, default='true',choices=['true','predicted']) 
parser.add_argument('--k_fold_splits', type=int, default=5)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--feature_extract', type=str, default= 'uni', choices=['uni','resnet'])

parser.add_argument('--base_input_path', type=str, default="/shares/sinha/mchoudhury/projects/pre-cancer-image-omics/TCGA_all_cancer_DeepPT",
                    help="base input path where all data is kept")
parser.add_argument('--output_path', type=str, default='/shares/sinha/sadeleye/ecPATH_Results/ecdna_LabelResult')

parser.add_argument('--corr_threshold', type=int, default=0.4) #DeepPT correlation threshold
parser.add_argument('--q_value_threshold', type=int, default=0.05) #DeepPT p adj threshold
parser.add_argument('--top_gene_features', type=int, default=150) 

if __name__ == '__main__':
    print("Starting....")
    args = parser.parse_args()
    cancer_type = args.cancer_type
    ml_method = args.ml_methods  # LR, GB, SVM, RF (logistic regression, gradient boosting, SVM, RF)
    expression_type = args.expression_type  # "true" or "predicted"
    k_fold_splits = args.k_fold_splits  # usually 5
    repeats = args.epochs  # usually 20 so that there 5*20=100 training loops
    exp_model = args.feature_extract

    corr_threshold = args.corr_threshold  # DeepPT correlation threshold
    q_value_threshold = args.q_value_threshold  # DeepPT p adj threshold
    number_top_features = args.top_gene_features

    random.seed(42)

    # Set random seed for numpy
    np.random.seed(42)

    # output predictions file:
    if expression_type == "true":
        prediction_result_fn = args.output_path + f"/TCGA_{cancer_type}_{ml_method}_method_mean_ecDNA_predictions_nested_{k_fold_splits}_fold_{repeats}_repeat_on_{expression_type}_ex_DeepPT.csv"
        # prediction_result_fn = f"../results/TCGA_{cancer_type}_{ml_method}\
        # _method_mean_ecDNA_predictions_nested_{k_fold_splits}_fold_{repeats}\
        # _repeat_on_{expression_type}_ex_DeepPT.csv"
    if expression_type == "predicted":
        prediction_result_fn = args.output_path + f"/TCGA_{cancer_type}_{ml_method}_method_mean_ecDNA_predictions_nested_{k_fold_splits}_fold_{repeats}_repeat_on_{expression_type}_ex_DeepPT_{exp_model}.csv"
        
    ## output models and genes for models files:
    output_model_file_fn = args.output_path + f"/TCGA_ecDNA_models/TCGA_{cancer_type}\_{k_fold_splits}_split_{repeats}_repeat_{expression_type}_expression_{ml_method}_models_{exp_model}.pkl"
    output_gene_feature_file_fn = args.output_path + f"/TCGA_ecDNA_models/TCGA_{cancer_type}_{k_fold_splits}_split_{repeats}_repeat_{expression_type}_expression_{ml_method}_models_gene_features_{exp_model}.pkl"
    output_gene_feature_csv_file_fn = args.output_path + f"/TCGA_ecDNA_models/TCGA_{cancer_type}_{k_fold_splits}_split_{repeats}_repeat_{expression_type}_expression_{ml_method}_models_gene_features_{exp_model}.csv"

    #Read expression file and ecDNA file
    true_expression_df = pd.read_csv( args.base_input_path 
        + f"/data/TCGA_{cancer_type}.DeepPT_normalized_reformatted_true_expression_per_sample_{exp_model}.txt",
        delimiter='\t')
    predicted_expression_df = pd.read_csv(args.base_input_path + f"/data/TCGA_{cancer_type}.DeepPT_reformatted_avg_predicted_expression_per_sample_"
                            + f"{exp_model}.txt", delimiter='\t')  # reorder the samples here
    ecDNA_df = pd.read_csv(args.base_input_path + '/../TCGA_eCDNA/data/TCGA_tumor_samples_all_cancer_type_ecDNA_and_other_variant_status.csv')
    print(ecDNA_df)

    # Gene info file
    genes_info_file = args.base_input_path + f"/results/TCGA_{cancer_type}.DeepPT_pred_coef_pvalue_per_gene_{exp_model}.txt"

    print("Loading data...")
    model_results, merged_df_pred, genes_of_interest = data_prep(cancer_type, exp_model, corr_threshold, q_value_threshold,args.output_path,genes_info_file,true_expression_df,predicted_expression_df,ecDNA_df)
    print("Data Loaded \n Now Starting to Train Model... ")
    auc_scores, model_pred_dfs = train( model_results, repeats, k_fold_splits, genes_of_interest,
                                        expression_type, ml_method, number_top_features,
                                        output_model_file_fn, 
                                        output_gene_feature_file_fn,
                                        output_gene_feature_csv_file_fn
                                    )
    print("Training Finished. Saving Results..")
    predict_and_results(auc_scores, model_pred_dfs, prediction_result_fn, merged_df_pred)
    print(f"Results saved at {args.output_path}")
