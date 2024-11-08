import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")

MODE = "prediction"  # or "reviewer_test" for testing (skip preprocessing & use test features)

basic_param = {
    "input_dir": os.path.join(BASE_DIR, "input"),
    "output_dir": os.path.join(BASE_DIR, "output"),
    "cancer_type": "GBM",  # only support ['LGG', 'GBM', 'STAD']
}

data_cloud_param = {
    "gcloud_drive": "",
    "zenodo_record_id": "14057816",  # 11/08/2024
}

preprocess_param = {
    "verbose": True,
    "slide_extention": ".svs",
    "input_keyword": "TCGA",  # ecPATH will automatically find the input files by matching this keyword, example, "test": test_1.svs, test_2.svs, ..., test_n.svs
    "generate_mask": True,
    "mag_selected": 20,
    "mag_assumed": 40,
    "evaluate_edge": True,
    "tile_size": 512,
    "mask_downsampling": 16,
    "edge_mag_thrsh": 15,
    "edge_fraction_thrsh": 0.5,
    "model_tile_size": 224,
}

feature_extraction_param = {
    "pretrained_model_name": "resnet",  # or UNI, case insensitive
    "model_dir": os.path.join(BASE_DIR, "assets"),
    "pretrained_model_param": {
        "resnet50": {},
        "UNI": {
            "login_token": ""
        },  # 'your_login_token' with approved access from huggingface.co
    },
}

ecDNA_param = {
    # Youden J statistic threshold for ecDNA prediction
    "threshold": {"LGG": 0.268, "GBM": 0.226, "STAD": 0.567},
}
