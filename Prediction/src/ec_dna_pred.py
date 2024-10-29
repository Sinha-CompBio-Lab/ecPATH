import os
import pickle

import joblib
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.linear_model import LogisticRegression


class ecDNA_Predictor:
    def __init__(self, input_predictions, model_path, model_name, cancer_type):
        self.input_predictions = input_predictions
        self.model_name = model_name
        self.cancer_type = cancer_type
        self.model_path = os.path.join(
            model_path,
            model_name.lower(),
            f"TCGA_{cancer_type}_5_split_200_repeat_predicted_expression_LR_models_{model_name.lower()}.pkl",
        )
        self.predictor_path = os.path.join(
            model_path,
            model_name.lower(),
            f"TCGA_{cancer_type}_5_split_200_repeat_predicted_expression_LR_models_gene_features_{model_name.lower()}.pkl",
        )
        self.model = None
        self.predictors = None

    def predict(self):
        self.__load_model()
        self.__load_predictors()
        preds = []
        for i, model in enumerate(self.model):
            temp_model_predictor = (self.predictors.iloc[i]).tolist()
            temp_predictors_array = self.input_predictions[temp_model_predictor].values
            y_pred_proba = model.predict_proba(temp_predictors_array)[:, 1]
            preds.append(y_pred_proba)
        return preds

    def __load_model(self):
        self.model = joblib.load(self.model_path)

    def __load_predictors(self):
        self.predictors = pd.read_pickle(self.predictor_path)
