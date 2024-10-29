import os

import numpy as np
import pandas as pd
import torch
from src.MLP import MLP_regression


class GeneExpressionPredictor:
    def __init__(
        self, input_files, model_path, model_name, cancer_type, slide_extention, device
    ):
        self.input_files = input_files
        self.feature_list = []
        self.model_name = model_name
        self.model_path = os.path.join(model_path, model_name.lower(), cancer_type)
        self.slide_extention = slide_extention
        self.device = device

    def load_features(self):
        for file in self.input_files:
            feature_path = os.path.join(
                file.replace(self.slide_extention, ""),
                "_features",
                f"features_{self.model_name.lower()}.npy",
            )
            self.feature_list.append(np.load(feature_path))

    def predict(self):
        preds_all_sample = []
        for feature in self.feature_list:
            feature_tensor = torch.Tensor(feature).float().to(self.device)
            outer_loop_preds = {i: [] for i in range(5)}  # 5 outer loops

            for i in range(25):
                out_loop, inner_loop = divmod(i, 5)
                genes_to_predict, model = self.__load_model(i)

                model.eval()
                with torch.no_grad():
                    pred = model.forward(feature_tensor.float().to(self.device))
                    pred_numpy = pred.detach().cpu().numpy()

                    outer_loop_preds[out_loop].append(pred_numpy)

            feature_means = []
            for out_loop in range(5):
                inner_preds = outer_loop_preds[out_loop]
                inner_mean = np.mean(inner_preds, axis=0)
                feature_means.append(inner_mean)

            final_mean = np.mean(feature_means, axis=0)
            preds_all_sample.append(final_mean)
        self.predictions = preds_all_sample

    def __load_model(self, fold):
        # decide the input size based on the model
        if "uni" in self.model_name.lower():
            n_input = 1024
        elif "resnet" in self.model_name.lower():
            n_input = 2048
        else:
            raise ValueError("Invalid model name, must be 'uni' or 'resnet'")

        # decide the number of genes to predict
        genes_to_predict = pd.read_csv(
            os.path.join(self.model_path, f"fold_{fold}", "genes_to_predict.txt"),
            header=1,
        ).values.flatten()
        self.genes_to_predict = genes_to_predict

        # load the model
        model = MLP_regression(
            n_inputs=n_input,
            n_hiddens=512,
            n_outputs=len(genes_to_predict),
            dropout=0.2,
            bias_init=None,
        )
        model.to(self.device)
        model.load_state_dict(
            torch.load(
                os.path.join(self.model_path, f"fold_{fold}", "model_trained.pth"),
                map_location=self.device,
            )
        )
        return genes_to_predict, model

    def get_predictions_df(self):
        predictions = pd.DataFrame(self.predictions)
        predictions.columns = self.genes_to_predict
        return predictions
