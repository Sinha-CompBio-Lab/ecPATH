import os

import cv2
import numpy as np
import timm
import torch
import torchvision
import torchvision.transforms as transforms
from huggingface_hub import login
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch import nn
from torchvision.models import resnet50


class LoadTVmodels(nn.Module):
    def __init__(self, weights_dir=None):
        super().__init__()
        if weights_dir:
            os.makedirs(weights_dir, exist_ok=True)
            os.environ["TORCH_HOME"] = weights_dir
        self.resnet = resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
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


class LoadHFModel:
    def __init__(self, token, model_name, weights_dir=None):
        self.token = token
        self.model_name = model_name
        if weights_dir:
            os.makedirs(weights_dir, exist_ok=True)
            os.environ["TORCH_HOME"] = weights_dir
        if self.token is None:
            raise ValueError("No token provided.")
        login(token=self.token)

    def load_model(self):

        try:
            model = timm.create_model(
                self.model_name,
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=True,
            )
            transform = create_transform(
                **resolve_data_config(model.pretrained_cfg, model=model)
            )
            return model, transform

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None


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


def init_random_seed(random_seed=42):
    # Python RNG
    np.random.seed(random_seed)

    # Torch RNG
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
