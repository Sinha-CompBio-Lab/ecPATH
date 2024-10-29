import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Keyword (input file name)

basic_param = {
    "input_dir": os.path.join(BASE_DIR, "input"),
    "output_dir": os.path.join(BASE_DIR, "output"),
    "cancer_type": "LGG",
}

preprocess_param = {
    "verbose": True,
    "slide_extention": ".svs",
    "input_keyword": "test",  # ecPATH will automatically find the input files by matching this keyword, example, "test": test_1.svs, test_2.svs, ..., test_n.svs
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
    "pretrained_model_name": "resnet",  # UNI, case insensitive
    "model_dir": os.path.join(BASE_DIR, "assets"),
    "pretrained_model_param": {
        "resnet50": {},
        "UNI": {
            "login_token": ""
        },  # 'your_login_token' with approved access from huggingface.co
    },
}
