import os
import pickle
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.stats.multitest as smt
import torch
import h5py
import torch.nn.functional as F
from scipy.stats import norm, pearsonr
from torch.utils.data import Dataset, Subset
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, GroupKFold, StratifiedGroupKFold, GroupShuffleSplit, train_test_split
from collections import defaultdict

##===================================================================================================
class Feature_Dataset_GenePrdt(Dataset):
    def __init__(self,filepaths, targets):
        """
        Args:
        file_path (string): Path to .npy file containing slide feature data.
        """
        self.features = [np.load(temp_feature).astype(np.float32) for temp_feature in filepaths]
        self.targets = [
            (np.array(genes, dtype=np.float32), np.array(status, dtype=np.float32))
            for genes, status in targets
        ]
        # self.targets = np.array(targets, dtype=np.float32)
        self.length = len(self.features)

    def __len__(self):
        return self.length

    def __getitem__(self,idx):
        sample = torch.Tensor(self.features[idx]).float()
        target_gene = torch.Tensor(self.targets[idx][0]).float()
        target_status = torch.tensor([self.targets[idx]][1], dtype=torch.float)
        return sample, (target_gene, target_status)

class Feature_Dataset(Dataset):
    def __init__(self,filepaths, targets, ext):
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
     

def get_detailed_metrics(model, dataset, batch_size=None):
    """
    Get detailed evaluation metrics for final model assessment
    
    Parameters:
    -----------
    model : nn.Module
    dataset : Dataset
        input feature dataset to evaluate on
        
    Returns:
    --------
    dict
        Detailed performance metrics
    """
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
            y = y.view(1,1) # change y shave to be 1 x 1 vector
            
            # Calculate loss
            loss = loss_fn(pred, y.float().to(device))
            total_loss += loss.item()
            
            # Convert to binary predictions
            binary_pred = (pred.cpu() >= 0.5).float()
            
            # Store results
            all_labels.append(y.cpu().numpy())
            all_preds.append(binary_pred.numpy())
            all_probs.append(pred.cpu().numpy())
    
    # Convert lists to numpy arrays and flatten
    all_labels = np.array(all_labels).flatten()
    all_preds = np.array(all_preds).flatten()
    all_probs = np.array(all_probs).flatten()
    
    # Calculate average loss
    avg_loss = total_loss / len(dataset)
    
    # Calculate metrics
    metrics = {
        'loss': avg_loss,
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'accuracy': accuracy_score(all_labels, all_preds)
    }
    
    # Add AUC if we have both classes
    if len(np.unique(all_labels)) > 1:
        metrics['auc'] = roc_auc_score(all_labels, all_probs)
    else:
        metrics['auc'] = 0.5
    
    return metrics


def create_stratified_grouped_cv_splits(dataset_indices, y, group_ids, n_splits=5, random_state=42):
    """
    Create cross-validation splits where:
    1. Samples from the same group stay together
    2. Class distribution is preserved in each fold
    
    Parameters:
    -----------
    dataset_indices : array-like
        Indices of the dataset to split
    y : array-like
        Target labels (used for stratification)
    group_ids : array-like
        Group identifiers (e.g., patient IDs) for each sample
    n_splits : int
        Number of folds
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    list of tuples
        List of (train_indices, test_indices) for each fold
    """
    # Ensure arrays are numpy arrays
    dataset_indices = np.array(dataset_indices)
    y = np.array(y)
    group_ids = np.array(group_ids)
    
    # Use sklearn's StratifiedGroupKFold
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Get the splits
    splits = []
    for train_idx, test_idx in cv.split(dataset_indices, y, groups=group_ids):
        # Convert to actual indices from the dataset
        train_indices = dataset_indices[train_idx]
        test_indices = dataset_indices[test_idx]
        splits.append((train_indices, test_indices))
    
    return splits


def create_train_val_test_split(dataset_indices, y, group_ids, test_size=0.2, val_size=0.1, random_state=42):
    """
    Create train/validation/test splits where:
    1. Samples from the same group stay together
    2. Class distribution is preserved in each split
    
    Parameters:
    -----------
    dataset_indices : array-like
        Indices of the dataset to split
    y : array-like
        Target labels (used for stratification)
    group_ids : array-like
        Group identifiers (e.g., patient IDs) for each sample
    test_size : float
        Proportion of the dataset to include in the test split (default 0.2)
    val_size : float
        Proportion of the training set to include in the validation split (default 0.1)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (train_indices, val_indices, test_indices)
    """
    # Ensure arrays are numpy arrays
    dataset_indices = np.array(dataset_indices)
    y = np.array(y)
    group_ids = np.array(group_ids)
    
    # Step 1: First split the data into train+val (80%) and test (20%) sets
    # We'll use GroupShuffleSplit to maintain group integrity while splitting
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    # Get the indices for the train+val and test sets
    train_val_idx, test_idx = next(gss.split(dataset_indices, y, groups=group_ids))
    
    # Get the actual indices from the dataset
    train_val_indices = dataset_indices[train_val_idx]
    test_indices = dataset_indices[test_idx]
    
    # Get corresponding labels and group IDs for the train+val set
    train_val_y = y[train_val_idx]
    train_val_groups = group_ids[train_val_idx]
    
    # Step 2: Split the train+val set into train and validation
    # Calculate the validation size relative to the train+val set
    # If val_size is 0.1 of the whole dataset and we have 80% in train+val, 
    # then val_size should be 0.1/0.8 = 0.125 of the train+val set
    effective_val_size = val_size / (1 - test_size)
    
    # Use another GroupShuffleSplit to maintain group integrity
    gss_val = GroupShuffleSplit(n_splits=1, test_size=effective_val_size, random_state=random_state)
    
    # Get the indices for the train and validation sets
    train_idx, val_idx = next(gss_val.split(train_val_indices, train_val_y, groups=train_val_groups))
    
    # Get the actual indices from the dataset
    train_indices = train_val_indices[train_idx]
    val_indices = train_val_indices[val_idx]
    
    # Verify the class distributions in each split
    print(f"Total samples: {len(dataset_indices)}")
    print(f"Train samples: {len(train_indices)} ({len(train_indices)/len(dataset_indices):.2%})")
    print(f"Validation samples: {len(val_indices)} ({len(val_indices)/len(dataset_indices):.2%})")
    print(f"Test samples: {len(test_indices)} ({len(test_indices)/len(dataset_indices):.2%})")
    
    for label in np.unique(y):
        total_count = np.sum(y == label)
        train_count = np.sum(y[np.isin(dataset_indices, train_indices)] == label)
        val_count = np.sum(y[np.isin(dataset_indices, val_indices)] == label)
        test_count = np.sum(y[np.isin(dataset_indices, test_indices)] == label)
        
        print(f"Label {label}:")
        print(f"  Train: {train_count}/{total_count} ({train_count/total_count:.2%})")
        print(f"  Val: {val_count}/{total_count} ({val_count/total_count:.2%})")
        print(f"  Test: {test_count}/{total_count} ({test_count/total_count:.2%})")
    
    return train_indices, val_indices, test_indices


def create_grouped_cv_splits(dataset_indices, group_ids, n_splits=5, random_state=42):
    """
    Create cross-validation splits where samples from the same individual stay together.
    
    Parameters:
    -----------
    dataset_indices : list
        Indices of the dataset to split
    group_ids : list
        Group identifiers (e.g., patient IDs) for each sample in dataset_indices
    n_splits : int
        Number of folds
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    list of tuples
        List of (train_indices, test_indices) for each fold
    """
    # Ensure group_ids is a numpy array
    group_ids = np.array(group_ids)
    
    # Use GroupKFold to keep samples from the same group in the same fold
    group_kfold = GroupKFold(n_splits=n_splits)
    
    # Get the splits
    splits = []
    for train_idx, test_idx in group_kfold.split(dataset_indices, groups=group_ids,random_state=random_state):
        # Convert to actual indices from the dataset
        train_indices = dataset_indices[train_idx]
        test_indices = dataset_indices[test_idx]
        splits.append((train_indices, test_indices))
    
    return splits

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
