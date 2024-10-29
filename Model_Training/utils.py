import os
import pickle
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.stats.multitest as smt
import torch
import torch.nn.functional as F
from scipy.stats import norm, pearsonr
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import Dataset, Subset


##===================================================================================================
#### Build dataset
class slide_target_dataset(Dataset):
    ## input: features_list[n_slides](slide_name, features[n_tiles,n_features])
    ## target[n_slides, n_target]

    def __init__(self, features, targets):

        self.features = [
            (slide_name, np.array(feat, dtype=np.float32))
            for slide_name, feat in features
        ]
        self.targets = np.array(targets, dtype=np.float32)
        self.dim = self.features[0][1].shape[1]  ## n_features

    def __getitem__(self, index):
        sample = torch.Tensor(self.features[index][1]).float()
        target = torch.Tensor(self.targets[index]).float()

        # if target.dim() == 1:
        #     target = target.unsqueeze(0)

        return sample, target

    def __len__(self):
        return len(self.features)


##===================================================================================================
def load_dataset(path2features, path2target, path2split, ik_fold, il_fold, target_cols):

    ## load image feature
    if path2features.endswith(".npy"):
        features = np.load(path2features, allow_pickle=True)
    else:
        with open(path2features, "rb") as f:
            features = pickle.load(f)

    print("len(features):", len(features))

    ## load target
    if path2target.endswith(".csv"):
        df_target = pd.read_csv(path2target, index_col=None, usecols=target_cols)[
            target_cols
        ]
    else:
        df_target = pd.read_pickle(path2target)

    targets = df_target[target_cols].values
    print("targets.shape:", targets.shape)

    ## create dataset
    dataset = slide_target_dataset(features, targets)

    ## load_train_valid_test_idx:
    #### Mudra's addition as she saved as a pickle file
    with open(path2split, "rb") as file:
        train_all_idx, valid_all_idx, test_all_idx = pickle.load(file)

    train_idx = train_all_idx[ik_fold][il_fold]
    valid_idx = valid_all_idx[ik_fold][il_fold]
    test_idx = test_all_idx[ik_fold]

    ### Tai's version as he saved as a npz file
    # train_valid_test_idx = np.load(path2split, allow_pickle=True)
    #
    # train_idx = train_valid_test_idx["train_idx"][ik_fold][il_fold]
    # valid_idx = train_valid_test_idx["valid_idx"][ik_fold][il_fold]
    # test_idx = train_valid_test_idx["test_idx"][ik_fold]

    ## split train, valid, test dataset
    train_set = Subset(dataset, train_idx)
    valid_set = Subset(dataset, valid_idx)
    test_set = Subset(dataset, test_idx)

    return train_set, valid_set, test_set


##===================================================================================================
def compute_coefs(labels, preds):
    return np.array(
        [pearsonr(labels[:, i], preds[:, i])[0] for i in range(labels.shape[1])]
    )


def compute_slope(labels, preds):
    return np.array(
        [np.polyfit(labels[:, i], preds[:, i], 1)[0] for i in range(labels.shape[1])]
    )


# def compute_coef_slope(labels, preds):

#     coef = np.array(
#         [pearsonr(labels[:, i], preds[:, i])[0] for i in range(labels.shape[1])]
#     )
#     slope = np.array(
#         [np.polyfit(labels[:, i], preds[:, i], 1)[0] for i in range(labels.shape[1])]
#     )

#     return coef, slope


def compute_coef_slope(labels, preds):
    coef = []
    slope = []

    for i in range(labels.shape[1]):
        label_col = labels[:, i]
        pred_col = preds[:, i]

        if np.any(np.isnan(label_col)) or np.any(np.isnan(pred_col)):
            print(f"Found nan in labels or predictions for gene {i}")
            coef.append(np.nan)
            slope.append(np.nan)
            continue

        # if np.std(label_col) == 0 or np.std(pred_col) == 0:
        #     print(f"Zero variance in labels or predictions for gene {i}")
        #     coef.append(np.nan)
        #     slope.append(np.nan)
        #     continue

        if np.std(label_col) == 0:
            print(f"Zero variance in labels for gene {i}")
            coef.append(np.nan)
            slope.append(np.nan)
            continue

        if np.std(pred_col) == 0:
            print(f"Zero variance in predictions for gene {i}")
            coef.append(np.nan)
            slope.append(np.nan)
            continue

        try:
            coef.append(pearsonr(label_col, pred_col)[0])
        except Exception as e:
            print(f"Error computing pearsonr for gene {i}: {e}")
            coef.append(np.nan)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", np.RankWarning)
                poly_fit = np.polyfit(label_col, pred_col, 1)
                slope.append(poly_fit[0])
        except np.RankWarning:
            print(f"Polyfit poorly conditioned for gene {i}")
            slope.append(np.nan)
        except Exception as e:
            print(f"Error computing polyfit for gene {i}: {e}")
            slope.append(np.nan)

    return np.array(coef), np.array(slope)


def compute_f1_auc(labels, logits):
    labels = labels.numpy()
    probabilities = F.softmax(logits, dim=-1)
    print("Probabilities:\n", probabilities)
    # print("Sum of probabilities for each set of class scores:\n", probabilities.sum(dim=-1))
    if probabilities.dim() > 2 and probabilities.shape[1] == 1:
        probabilities = probabilities.squeeze(1)

    predictions = torch.argmax(probabilities, dim=1)
    print("Predictions:", predictions)
    # predictions = predictions.numpy()
    # probabilities = probabilities.numpy()

    # Calculate F1 scores
    f1_micro = f1_score(labels, predictions, average="micro")
    f1_macro = f1_score(labels, predictions, average="macro")
    f1_weighted = f1_score(labels, predictions, average="weighted")

    # Calculate AUC
    auc_score = roc_auc_score(labels, probabilities, multi_class="ovr")
    # print(auc_score)
    return f1_macro, f1_micro, f1_weighted, auc_score, predictions


def compute_f1_auc_binary(labels, probabilities, threshold):
    labels = labels.numpy()
    probabilities = probabilities.numpy()
    # probabilities = F.softmax(logits, dim=-1)
    # print("Probabilities:\n", probabilities)
    # print("Sum of probabilities for each set of class scores:\n", probabilities.sum(dim=-1))
    # if probabilities.dim() > 2 and probabilities.shape[1] == 1:
    #    probabilities = probabilities.squeeze(1)

    # predictions = torch.argmax(probabilities, dim=1)
    print(probabilities)
    predictions = (probabilities > threshold).astype(int)
    print("Predictions:", predictions)
    # predictions = predictions.numpy()
    # probabilities = probabilities.numpy()

    # Calculate F1 scores
    f1 = f1_score(labels, predictions)

    # Calculate AUC
    auc = roc_auc_score(labels, predictions)
    # print(auc_score)
    return f1, auc, predictions


##------------------------------------------------------------------
## R and p_1side values
def pearson_r_and_p(label, pred):
    R, p = pearsonr(label, pred)
    if R > 0:
        p_1side = p / 2.0
    else:
        p_1side = 1 - p / 2

    return p_1side


##------------------------------------------------------------------
## number of genes with Holm-Sidak correlated p-val<0.05
def number_predictable_genes(labels, preds):
    p = np.array(
        [pearson_r_and_p(labels[:, i], preds[:, i]) for i in range(preds.shape[1])]
    )

    return np.sum(p < 0.05)


def holm_sidak_p(labels, preds):
    return np.array(
        [pearson_r_and_p(labels[:, i], preds[:, i]) for i in range(preds.shape[1])]
    )


def compute_coef_slope_p(labels, preds):
    coef = np.array(
        [pearsonr(labels[:, i], preds[:, i])[0] for i in range(labels.shape[1])]
    )
    slope = np.array(
        [np.polyfit(labels[:, i], preds[:, i], 1)[0] for i in range(labels.shape[1])]
    )
    p_value = np.array(
        [pearson_r_and_p(labels[:, i], preds[:, i]) for i in range(preds.shape[1])]
    )

    return coef, slope, p_value


def compute_coef_slope_padj(labels, preds):
    coef = np.array(
        [pearsonr(labels[:, i], preds[:, i])[0] for i in range(labels.shape[1])]
    )
    slope = np.array(
        [np.polyfit(labels[:, i], preds[:, i], 1)[0] for i in range(labels.shape[1])]
    )
    p_value = np.array(
        [pearson_r_and_p(labels[:, i], preds[:, i]) for i in range(preds.shape[1])]
    )

    p_adj = smt.multipletests(
        p_value, alpha=0.05, method="hs", is_sorted=False, returnsorted=False
    )[1]

    return coef, slope, p_adj


##===================================================================================================


def init_random_seed(random_seed=42):
    # Python RNG
    np.random.seed(random_seed)

    # Torch RNG
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mad(data, axis=None):
    return np.median(np.abs(data - np.median(data, axis)), axis)


##===================================================================================================
def visualize_attention_weights(result_dir, attentions, n_samples=5):
    # Save all attention weights
    attention_file = os.path.join(result_dir, "attention_weights.pkl")
    with open(attention_file, "wb") as f:
        pickle.dump(attentions, f)
    print(f"Attention weights saved to {attention_file}")

    # Visualize a subset of attention weights
    n_samples = min(n_samples, len(attentions))
    fig, axs = plt.subplots(n_samples, 1, figsize=(10, 4 * n_samples))

    for i in range(n_samples):
        attention = attentions[i]
        ax = axs[i] if n_samples > 1 else axs
        ax.bar(range(len(attention)), attention)
        ax.set_title(f"Sample {i+1} Attention Weights")
        ax.set_xlabel("Tile Index")
        ax.set_ylabel("Attention Weight")

    plt.tight_layout()
    viz_file = os.path.join(result_dir, "attention_weights_viz.pdf")
    plt.savefig(viz_file, format="pdf", dpi=50)
    print(f"Attention weights visualization saved to {viz_file}")

    # Save summary statistics
    summary_stats = {
        "mean_attention": np.mean([att.mean() for att in attentions]),
        "std_attention": np.mean([att.std() for att in attentions]),
        "min_attention": min([att.min() for att in attentions]),
        "max_attention": max([att.max() for att in attentions]),
        "mean_num_tiles": np.mean([len(att) for att in attentions]),
    }

    summary_file = os.path.join(result_dir, "attention_summary.txt")
    with open(summary_file, "w") as f:
        for key, value in summary_stats.items():
            f.write(f"{key}: {value}\n")
    print(f"Attention summary statistics saved to {summary_file}")


##===================================================================================================
class ShuffledSubset(Dataset):
    def __init__(self, subset, seed=None):
        self.data = [item for item, _ in subset]
        self.labels = [label for _, label in subset]

        if seed is not None:
            random.seed(seed)
        random.shuffle(self.labels)  # Shuffle the labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


import cv2
import numpy as np
import torch
import torchvision
from torch import nn
from torchvision.models import resnet50


##======================================================================================================
class Feature_Extraction(nn.Module):
    def __init__(self, model_type="load_from_saved_file"):
        super().__init__()

        if model_type == "load_from_internet":
            self.resnet = resnet50(
                weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            )
        elif model_type == "load_from_saved_file":
            self.resnet = resnet50(weights=None)
        else:
            print(
                "cannot find model_type can only be load_from_internet or load_from_saved_file"
            )

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x


##======================================================================================================
def evaluate_tile_edge(img_np, edge_mag_thrsh, edge_fraction_thrsh):

    select = 1  ## initial value

    # img_np = np.array(img_RGB)
    tile_size = img_np.shape[0]

    ##---------------------------------------
    ## 0) exclude if edge_mag > 0.5
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Remove noise using a Gaussian filter
    # img_gray = cv2.GaussianBlur(img_gray, (5,5), 0)

    sobelx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0)
    sobely = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1)

    sobelx1 = cv2.convertScaleAbs(sobelx)
    sobely1 = cv2.convertScaleAbs(sobely)

    mag = cv2.addWeighted(sobelx1, 0.5, sobely1, 0.5, 0)

    unique, counts = np.unique(mag, return_counts=True)

    edge_mag = counts[np.argwhere(unique < edge_mag_thrsh)].sum() / (
        tile_size * tile_size
    )

    if edge_mag > edge_fraction_thrsh:
        select = 0

    return select


##======================================================================================================
def evaluate_tile_color(
    img_np,
    black_thrsh,
    black_pct_thrsh,
    blue_level_thrsh,
    red_level_thrsh,
    H_min,
    H_max,
    S_min,
    S_max,
    V_min,
    V_max,
    select,
):

    # img_np = np.array(img_RGB)

    L, A, B = cv2.split(cv2.cvtColor((img_np), cv2.COLOR_RGB2LAB))

    ##---------------------------------------
    ## 1) remove if percentage of black spot > 0.01
    black_pct = np.mean(L < black_thrsh)
    if black_pct > black_pct_thrsh:
        select = 0
        return select
    ##---------------------------------------
    ## 2) remove if too blue (heavy mark), or too red (blood)
    red, green, blue = (
        np.mean(img_np[:, :, 0]),
        np.mean(img_np[:, :, 1]),
        np.mean(img_np[:, :, 2]),
    )
    blue_level = blue / (red + green)
    blue_level2 = blue * blue_level

    if blue_level2 > blue_level_thrsh:
        select = 0
        return select

    ##---
    red_level = red / (green + blue)
    red_level2 = red * red_level

    if red_level2 > red_level_thrsh:
        select = 0
        return select

    ##---------------------------------------
    ## 3) remove if tile has the same color suggested (using color detection)
    H, S, V = cv2.split(cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV))
    H, S, V = np.mean(H), np.mean(S), np.mean(V)

    if (
        H_min <= H
        and H <= H_max
        and S_min <= S
        and S <= S_max
        and V_min <= V
        and V <= V_max
    ):
        select = 0
        return select

    return select


##================================================================================================
def init_random_seed(random_seed=42):
    # Python RNG
    np.random.seed(random_seed)

    # Torch RNG
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
