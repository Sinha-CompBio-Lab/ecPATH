import os
import sys

## SET UP PATH ##
BASE = os.path.join("/shares", "sinha", "lliu", "projects", "pre-cancer-image-omics")

cur_type = sys.argv[1]  # "LUSC"

cur_slide_item = os.path.join(
    sys.argv[2].strip(), ""
)  # 36ecd34f-4d71-4591-a6bc-9de584aa5dbf

# cur_slide_item = "36ecd34f-4d71-4591-a6bc-9de584aa5dbf"


target_slide_dir = os.path.join(BASE, "rawData", "slides", cur_type, cur_slide_item)
# path2model = os.path.join(BASE, "srcRef", "ResNet50_IMAGENET1K_V2.pt")

path2mask = os.path.join(target_slide_dir, "_mask/", "")
path2features = os.path.join(target_slide_dir, "_features", "")
path2coordinates = os.path.join(target_slide_dir, "_coordinates", "")
path2tiles = os.path.join(target_slide_dir, "_tiles", "")
os.makedirs(path2mask, exist_ok=True)
os.makedirs(path2features, exist_ok=True)
os.makedirs(path2coordinates, exist_ok=True)
os.makedirs(path2tiles, exist_ok=True)

for item in os.listdir(target_slide_dir):
    if item.endswith(".svs"):
        target_slide_item = os.path.join(target_slide_dir, item)
        slide_name = item.replace(".svs", "")

if not os.path.exists(
    os.path.join(path2mask, f"{slide_name}-uni.pdf")
) or not os.path.exists(os.path.join(path2features, f"{slide_name}-uni.npy")):
    try:
        import concurrent.futures
        import os
        import platform
        import sys
        import threading
        import time
        from os.path import join as j_

        import matplotlib.pyplot as plt
        import numpy as np
        import openslide
        import pandas as pd
        import timm
        import torch
        import torchvision
        import torchvision.transforms as transforms
        import utils_color_norm as utils_color_norm
        from openslide import OpenSlide
        from PIL import Image
        from torchvision import transforms
        from transformers import ViTImageProcessor, ViTModel

        # loading all packages here to start
        from UNI.uni import get_encoder
        from UNI.uni.downstream.eval_patch_features.fewshot import (
            eval_fewshot,
            eval_knn,
        )
        from UNI.uni.downstream.eval_patch_features.linear_probe import (
            eval_linear_probe,
        )
        from UNI.uni.downstream.eval_patch_features.metrics import (
            get_eval_metrics,
            print_metrics,
        )
        from UNI.uni.downstream.eval_patch_features.protonet import (
            ProtoNet,
            prototype_topk_vote,
        )
        from UNI.uni.downstream.extract_patch_features import (
            extract_patch_features_from_dataloader,
        )
        from UNI.uni.downstream.utils import concat_images
        from utils_preprocessing import *

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("device:", device)
        init_random_seed(random_seed=42)
        color_norm = utils_color_norm.macenko_normalizer()

        ##---------------------------------------
        ## hyper-parameters
        mag_assumed = 40
        evaluate_edge = True
        extract_pretrained_features = True
        save_tile_file = False
        data_augmentation = True  # False

        mag_selected = 20
        tile_size = 512  # why?
        mask_downsampling = 16  # for visualization purpose?

        ## evaluate tile
        edge_mag_thrsh = 15
        edge_fraction_thrsh = 0.5

        model_tile_size = 224
        batch_size = 64

        mask_tile_size = int(np.ceil(tile_size / mask_downsampling))
        platform_uname = platform.uname()

        # open slides
        slide = OpenSlide(target_slide_item)
        cur_slide_info = pd.Series(slide.properties)

        ## magnification check
        if openslide.PROPERTY_NAME_OBJECTIVE_POWER in slide.properties:
            mag_max = slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
            print("mag_max:", mag_max)
            mag_original = mag_max
        else:
            print("[WARNING] mag not found, assuming: {mag_assumed}")
            mag_max = mag_assumed
            mag_original = 0

        ## downsample_level
        downsampling = int(int(mag_max) / mag_selected)
        print(f"downsampling: {downsampling}")

        ## slide size at largest level (level=0)
        px0, py0 = slide.level_dimensions[0]
        tile_size0 = int(tile_size * downsampling)
        # print(f"px0: {px0}, py0: {py0}, tile_size0: {tile_size0}")

        n_rows, n_cols = int(py0 / tile_size0), int(px0 / tile_size0)
        # print(f"n_rows: {n_rows}, n_cols: {n_cols}")

        n_tiles_total = n_rows * n_cols
        # print(f"n_tiles_total: {n_tiles_total}")

        ## CREATE MASKS ##
        img_mask = np.full(
            (int((n_rows) * mask_tile_size), int((n_cols) * mask_tile_size), 3), 255
        ).astype(np.uint8)
        mask = np.full(
            (int((n_rows) * mask_tile_size), int((n_cols) * mask_tile_size), 3), 255
        ).astype(np.uint8)

        ##======================================================================================================
        def process_tile(
            row,
            col,
            i_tile,
            slide,
            tile_size0,
            tile_size,
            mask_tile_size,
            edge_mag_thrsh,
            edge_fraction_thrsh,
            color_norm,
            save_tile_file,
            downsampling,
        ):
            tile = slide.read_region(
                (col * tile_size0, row * tile_size0),
                level=0,
                size=[tile_size0, tile_size0],
            )  ## RGBA image
            tile = tile.convert("RGB")

            tile_info = (row, col, i_tile, None, None, None)

            if tile.size[0] == tile_size0 and tile.size[1] == tile_size0:
                # downsample to target tile size
                tile = tile.resize((tile_size, tile_size))

                mask_tile = np.array(tile.resize((mask_tile_size, mask_tile_size)))

                img_mask[
                    int(row * mask_tile_size) : int((row + 1) * mask_tile_size),
                    int(col * mask_tile_size) : int((col + 1) * mask_tile_size),
                    :,
                ] = mask_tile

                tile = np.array(tile)

                ## evaluate tile
                select = evaluate_tile_edge(tile, edge_mag_thrsh, edge_fraction_thrsh)

                if select == 1:
                    ## 2022.09.08: color normalization:
                    tile_norm = Image.fromarray(color_norm.transform(tile))

                    mask_tile_norm = np.array(
                        tile_norm.resize((mask_tile_size, mask_tile_size))
                    )

                    mask[
                        int(row * mask_tile_size) : int((row + 1) * mask_tile_size),
                        int(col * mask_tile_size) : int((col + 1) * mask_tile_size),
                        :,
                    ] = mask_tile_norm

                    # if save_tile_file:
                    #     tile_name = (
                    #         "tile_"
                    #         + str(row).zfill(5)
                    #         + "_"
                    #         + str(col).zfill(5)
                    #         + "_"
                    #         + str(i_tile).zfill(5)
                    #         + "_"
                    #         + str(downsampling).zfill(3)
                    #     )

                    #     tile_norm.save(f"{tile_folder}/{tile_name}.png")

                    tile_info = (row, col, i_tile, tile_norm, mask_tile_norm, mask_tile)

            return tile_info

        i_tile = 0
        tiles_list = []
        idx_list = []

        tile_kv_pair = {}
        col_list = []
        row_list = []
        i_tile_list = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=62) as executor:
            futures = []
            for row in range(n_rows):
                for col in range(n_cols):
                    futures.append(
                        (
                            row,
                            col,
                            i_tile,
                            executor.submit(
                                process_tile,
                                row,
                                col,
                                i_tile,
                                slide,
                                tile_size0,
                                tile_size,
                                mask_tile_size,
                                edge_mag_thrsh,
                                edge_fraction_thrsh,
                                color_norm,
                                save_tile_file,
                                downsampling,
                            ),
                        )
                    )
                    i_tile += 1

            for row, col, i_tile, future in futures:
                row, col, i_tile, tile_norm, mask_tile_norm, mask_tile = future.result()
                if tile_norm is not None:
                    idx_pair = (row, col)
                    idx_list.append(idx_pair)
                    tile_kv_pair[idx_pair] = tile_norm

                    # tiles_list.append(tile_norm)
                    col_list.append(col)
                    row_list.append(row)

                    i_tile_list.append(i_tile)

        tiles_list = []
        sorted_idx_list = sorted(idx_list)
        for item in sorted_idx_list:
            tiles_list.append(tile_kv_pair[item])

        ##======================================================================================================
        ## 2023.05.27: save tile coordinates:
        downsampling_list = [downsampling] * len(row_list)
        df_coordinates = pd.DataFrame(
            {
                "row": row_list,
                "col": col_list,
                "i_tile": i_tile_list,
                "downsampling": downsampling,
            }
        )
        df_coordinates.to_csv(
            f"{path2coordinates}{slide_name}-uni.csv", index_label="tile_idx"
        )

        ##======================================================================================================
        ## PLOT: draw color lines on the mask
        line_color = [0, 255, 0]

        n_tiles = len(tiles_list)

        img_mask[:, ::mask_tile_size, :] = line_color
        img_mask[::mask_tile_size, :, :] = line_color
        mask[:, ::mask_tile_size, :] = line_color
        mask[::mask_tile_size, :, :] = line_color

        fig, ax = plt.subplots(1, 2, figsize=(30, 15))
        ax[0].imshow(img_mask)
        ax[1].imshow(mask)

        ax[0].set_title(
            f"{slide_name}, mag_original: {mag_original}, mag_assumed: {mag_assumed}"
        )
        ax[1].set_title(
            f"n_rows: {n_rows}, n_cols: {n_cols}, n_tiles_total: {n_tiles_total}, n_tiles_selected: {n_tiles}"
        )

        plt.tight_layout(h_pad=0.4, w_pad=0.5)
        plt.savefig(f"{path2mask}{slide_name}-uni.pdf", format="pdf", dpi=50)
        plt.close()

        ##======================================================================================

        local_dir = os.path.join(
            BASE,
            "src",
            "deeppt",
            "UNI",
            "assets",
            "ckpts",
            "vit_large_patch16_224.dinov2.uni_mass100k",
        )  # "../assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/"

        model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )

        model.load_state_dict(
            torch.load(
                os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"
            ),
            strict=True,
        )

        model.eval()
        model.to(device)
        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        transform

        def extract_features_from_tiles_UNI(tiles_list):
            features = []

            # Apply transformation to each tile and move to GPU within the loop
            for tile in tiles_list:
                tile_temp = transform(tile).unsqueeze(dim=0).to(device)

                # Use inference mode for faster computations
                with torch.inference_mode():
                    feature_temp = model(
                        tile_temp
                    )  # Extracted features (torch.Tensor) with shape [1, 1024]
                    features.append(feature_temp.detach().cpu().numpy())

            # Concatenate all features into a single NumPy array
            features = np.concatenate(features, axis=0)

            return features

        ##-----------------------
        if extract_pretrained_features:
            features = extract_features_from_tiles_UNI(tiles_list)

            np.save(f"{path2features}{slide_name}-uni.npy", features)

        ##======================================================================================
        if data_augmentation:
            for k in [1, 2, 3]:
                rot_angle = k * 90
                print("rot_angle:", rot_angle)
                tiles_list_rot = []
                for i in range(n_tiles):
                    tile_rot = transforms.functional.rotate(tiles_list[i], rot_angle)
                    tiles_list_rot.append(tile_rot)

                    ## save tile rot
                    # tile_name = f"tile_rot{k}_" + str(i).zfill(5)
                    # tile_rot.save(f"{tile_folder}/{tile_name}.png")

                features_rot = extract_features_from_tiles_UNI(tiles_list_rot)
                print("features_rot.shape:", features_rot.shape)
                np.save(f"{path2features}{slide_name}_rot{k}-uni.npy", features_rot)

    except Exception as e:
        # Ensure the output directory exists
        # os.makedirs(target_slide_dir, exist_ok=True)
        # Create the full path to the error log file
        error_file_path = os.path.join(target_slide_dir, "error_log-uni.txt")
        # Capture and write any error message to a file in the given directory
        with open(error_file_path, "a", encoding="utf-8") as error_file:
            error_file.write(f"An error occurred: {e}\n")

else:
    print("exitst..")
