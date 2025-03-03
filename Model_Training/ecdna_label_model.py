
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import warnings
import numpy as np
import pandas as pd

#Define the select_features_based_on_training_data function for predicted expression of genes
#This function takes average univariate AUC for true and predicted expression and selects the top and bottom as features
def select_features_based_on_predicted_training_data(X_train_predicted_ex, X_train_true_ex, y_train, include_low_AUC, number_top_features,genes_of_interest):
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




#Build Random Forest model
def Random_Forest_model(X_train_selected, X_test_selected, y_train, y_test, test_idx, i, inner_splits, patients):
    print("--- Random forest...")
    grid_search_iterations = 50
    
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
 
    
    return y_pred, best_model, auc, pred_df




#Build Logistic Regression model
def Logistic_Regression_model(X_train_selected, X_test_selected, y_train, y_test, test_idx, i, inner_splits, patients):
    print("--- Logistic Regression...")
    
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
    return y_pred, best_model, auc, pred_df
