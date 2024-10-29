import concurrent.futures
from collections import defaultdict

import openslide
import pandas as pd
from openslide import OpenSlide
from PIL import Image
from src.utils.utils_color_norm import *
from src.utils.utils_preprocessing import *


class preprocess:
    # constructor
    def __init__(
        self,
        BASE_DIR,
        input_files,
        preprocess_param,
        feature_extraction_param,
        device,
        max_workers,
    ):
        self.base_dir = BASE_DIR
        self.input_files = input_files
        self.preprocess_param = preprocess_param
        self.feature_extraction_param = feature_extraction_param
        self.intermediate_dir = defaultdict(dict)
        self.tile_list = []
        self.feature_list = []
        self.device = device
        self.color_norm = macenko_normalizer()
        init_random_seed()
        self.preprocess_param["verbose"] = False
        self.max_workers = max_workers

    def set_up(self):
        # set up directory for intermediate information, features, masks, tile_coord, etc.
        for file in self.input_files:
            # get the base name of the file
            alias = os.path.basename(file).replace(
                self.preprocess_param["slide_extention"], ""
            )
            dir_name = os.path.dirname(file)
            temp_dir = os.path.join(dir_name, alias)
            self.intermediate_dir[file] = defaultdict(dict)

            for target_folder in ["_masks", "_features", "_coordinates"]:
                target_slide_dir = os.path.join(temp_dir, target_folder)
                os.makedirs(target_slide_dir, exist_ok=True)
                self.intermediate_dir[file][target_folder] = target_slide_dir

    def process_tiles(self):
        for file in self.input_files:
            # filter
            print(f"Processing {os.path.basename(file)}...")

            feature_save_path = os.path.join(
                str(self.intermediate_dir[file]["_features"]),
                f"features_{self.feature_extraction_param['pretrained_model_name'].lower()}.npy",
            )
            if os.path.exists(feature_save_path):
                print(f"Features already extracted for {os.path.basename(file)}.")
            else:
                print(f"Processing {os.path.basename(file)}...")
                self.__process_tiles_single(file)
                print(f"Done.")

                # feature extraction
                print(f"Feature eactaction on {os.path.basename(file)}...")
                self.__feature_extraction_single(file)
                print(f"Done.")

    def __process_tiles_single(self, slide_path):
        slide = OpenSlide(slide_path)
        cur_slide_info = pd.Series(slide.properties)

        ## magnification check
        if openslide.PROPERTY_NAME_OBJECTIVE_POWER in slide.properties:
            mag_max = slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
            if self.preprocess_param["verbose"]:
                print(f"{os.path.basename(slide_path)} magnification: {mag_max}")
            mag_original = mag_max
        else:
            if self.preprocess_param["verbose"]:
                print(
                    f'[WARNING] mag not found, assuming: {self.preprocess_param["mag_assumed"]}'
                )
            mag_max = self.preprocess_param["mag_assumed"]
            mag_original = 0

        ## downsample_level
        mask_tile_size = int(
            np.ceil(
                self.preprocess_param["tile_size"]
                / self.preprocess_param["mask_downsampling"]
            )
        )

        downsampling = int(int(mag_max) / self.preprocess_param["mag_selected"])
        if self.preprocess_param["verbose"]:
            print(f"downsampling: {downsampling}")

        ## slide size at largest level (level=0)
        px0, py0 = slide.level_dimensions[0]
        tile_size0 = int(self.preprocess_param["tile_size"] * downsampling)
        n_rows, n_cols = int(py0 / tile_size0), int(px0 / tile_size0)
        n_tiles_total = n_rows * n_cols

        ## CREATE MASKS ##
        img_mask = np.full(
            (int((n_rows) * mask_tile_size), int((n_cols) * mask_tile_size), 3), 255
        ).astype(np.uint8)
        mask = np.full(
            (int((n_rows) * mask_tile_size), int((n_cols) * mask_tile_size), 3), 255
        ).astype(np.uint8)

        ### filter each tile
        i_tile = 0
        tiles_list = []
        idx_list = []

        tile_kv_pair = {}
        col_list = []
        row_list = []
        i_tile_list = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = []
            for row in range(n_rows):
                for col in range(n_cols):
                    futures.append(
                        (
                            row,
                            col,
                            i_tile,
                            executor.submit(
                                self.__process_tile_multithread_helper,
                                row,
                                col,
                                i_tile,
                                slide,
                                tile_size0,
                                self.preprocess_param["tile_size"],
                                mask_tile_size,
                                self.preprocess_param["edge_mag_thrsh"],
                                self.preprocess_param["edge_fraction_thrsh"],
                                self.color_norm,
                                img_mask,
                                mask,
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

        if self.preprocess_param["verbose"]:
            print(f"Number of tiles selected: {len(tiles_list)}")

        # save tiles to the instance variable
        self.tile_list.append(tiles_list)

        #### save tile coord
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
            os.path.join(
                self.intermediate_dir[slide_path]["_coordinates"],
                "tile_coordinates.csv",
            ),
            index_label="tile_idx",
        )
        if self.preprocess_param["verbose"]:
            print(
                f'Tile coordinates saved to {os.path.join(self.intermediate_dir[slide_path]["_coordinates"], "tile_coordinates.csv")}'
            )

        #### save masked images for visualization
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
            f"{os.path.basename(slide_path)}, mag_original: {mag_original}, mag_assumed: {self.preprocess_param['mag_assumed']}"
        )
        ax[1].set_title(
            f"n_rows: {n_rows}, n_cols: {n_cols}, n_tiles_total: {n_tiles_total}, n_tiles_selected: {n_tiles}"
        )
        plt.tight_layout(h_pad=0.4, w_pad=0.5)
        plt.savefig(
            os.path.join(str(self.intermediate_dir[slide_path]["_masks"]), "mask.pdf"),
            format="pdf",
            dpi=50,
        )
        plt.close()
        if self.preprocess_param["verbose"]:
            print(
                f'Mask images saved to {os.path.join(self.intermediate_dir[slide_path]["_mask"], "mask.pdf")}'
            )

    def __process_tile_multithread_helper(
        self,
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
        img_mask,
        mask,
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

                tile_info = (row, col, i_tile, tile_norm, mask_tile_norm, mask_tile)

        return tile_info

    def __feature_extraction_single(self, slide_path):
        """Extract features from slides using either UNI or ResNet50 model, processing one tile at a time."""
        temp_tile_list = self.tile_list[-1]
        model_name = self.feature_extraction_param["pretrained_model_name"]
        feature_save_path = os.path.join(
            str(self.intermediate_dir[slide_path]["_features"]),
            f"features_{model_name.lower()}.npy",
        )
        # if features already extracted, skip
        if os.path.exists(feature_save_path):
            print(f"Features already extracted for {os.path.basename(slide_path)}.")
            return
        else:
            try:
                # Model initialization based on type
                if "uni" in model_name.lower():
                    uni_token = self.feature_extraction_param["pretrained_model_param"][
                        "UNI"
                    ]["login_token"]

                    UNI_model = LoadHFModel(
                        token=uni_token,
                        model_name="hf-hub:MahmoodLab/uni",
                        weights_dir=self.feature_extraction_param["model_dir"],
                    )
                    model, _ = UNI_model.load_model()
                elif "resnet" in model_name.lower():
                    model = LoadTVmodels(
                        weights_dir=self.feature_extraction_param["model_dir"]
                    )

                # model set up
                model.to(self.device)
                model.eval()

                # transformation setup
                transform = transforms.Compose(
                    [
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
                #
                features = []
                with torch.inference_mode():
                    for tile in temp_tile_list:
                        # Transform and process single tile
                        tile_tensor = transform(tile).unsqueeze(0).to(self.device)
                        feature = model(tile_tensor)
                        features.append(feature.detach().cpu().numpy())

                        # Clear GPU memory after each tile
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                # Concatenate and save features
                features = np.concatenate(features, axis=0)
                self.feature_list.append(features)

                # Save to disk
                save_path = os.path.join(
                    str(self.intermediate_dir[slide_path]["_features"]),
                    f"features_{model_name.lower()}.npy",
                )
                np.save(save_path, features)

            except Exception as e:
                print(f"Error during feature extraction: {e}")
                raise
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
