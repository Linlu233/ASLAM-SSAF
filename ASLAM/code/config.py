import argparse

from utils.train_utils import add_flags_from_config


config_args = {
    "training_config": {
        "model_variant": ("baseline", str, "model variant: baseline or ssmattn"),
        "runs": (1, int, "num of runs"),
        "lr": (0.001, float, "learning rate"),
        "epochs": (401, int, "training epochs"),
        "cuda": ("cuda:0", str, "torch device"),
        "root": ("../../datasets", str, "root directory of dataset storage"),
        "device": (0, int, "legacy unused device id"),
        "wd": (5e-4, float, "weight decay"),
        "bs": (32, int, "batch size"),
        "patience": (20, int, "early stop patience"),
        "K": (20, int, "the K of Hits@K"),
        "metric": ("auc_ap", str, "performance metric to use"),
        "dynamic_train": (False, bool, "whether to use dynamic train"),
        "dynamic_val": (False, bool, "whether to use dynamic val"),
        "dynamic_test": (False, bool, "whether to use dynamic test"),
        "update_steps": (5, int, "steps to update semantic graph structures"),
        "fusion_dropout": (0.1, float, "dropout used by the improved fusion model"),
        "ssm_hidden": (32, int, "hidden dimension used by the improved selective SSM fusion"),
    },
    "data_config": {
        "dataset": ("citeseer", str, "which dataset to use"),
        "val_ratio": (0.05, float, "the ratio of val edges"),
        "test_ratio": (0.10, float, "the ratio of test edges"),
        "train_percent": (1.0, float, "the ratio of links of the split edges"),
        "val_percent": (1.0, float, "the ratio of links of the split edges"),
        "test_percent": (1.0, float, "the ratio of links of the split edges"),
        "use_new_split": (False, bool, "whether to use new split method of PyG"),
        "use_feat": (False, bool, "whether to use node features"),
    },
}

parser = argparse.ArgumentParser("Configs for Attribute and Structure Learning based on Attention Mechanism.")
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
