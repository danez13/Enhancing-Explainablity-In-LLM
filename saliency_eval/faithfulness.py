"""Computing Faithfulness measure for the saliency scores."""
import argparse
import os
import random
from argparse import Namespace
from functools import partial

import numpy as np
import torch
from sklearn.metrics import auc
from transformers import BertTokenizer

from models.train_cnn import eval_model
from models.data_loader import BucketBatchSampler, DatasetSaliency, collate_threshold, collate_nli, NLIDataset
from models.model_builder import CNN_MODEL


def get_model(model_path):
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)
    model_args = checkpoint['args']

    model_cp = CNN_MODEL(tokenizer, model_args,
                             n_labels=checkpoint['args']['labels']).to(device)

    model_cp.load_state_dict(checkpoint['model'])

    return model_cp, model_args


if __name__ == "__main__":
    args = {
        "gpu": False,
        "saliency": ["rand","shap", "sal_mean", "sal_l2", "occlusion_none", "lime", "inputx_mean", "inputx_l2", "guided_mean", "guided_l2"],
        "dataset": "snli",
        "dataset_dir": "data/e-SNLI/dataset",
        "test_saliency_dir": ["data/saliency/snli/cnn/","data/saliency/snli/random_cnn/"],
        "model_path": ["data/models/snli/cnn/cnn","data/models/snli/random_cnn/cnn"],
        "models_dir": ["data/models/snli/cnn/","data/models/snli/random_cnn/"],
        "model": "cnn"
    }

    device = torch.device("cuda") if args["gpu"] else torch.device("cpu")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    thresholds = list(range(0, 110, 10))
    aucs = []

    coll_call = collate_nli


    eval_fn = eval_model

    for saliency in args["saliency"]:
        for models_dir,test_saliency_dir in zip(args["models_dir"],args["test_saliency_dir"]):
            for model_path in os.listdir(models_dir):
                if model_path.endswith('.predictions'):
                    continue
                print('Model', model_path, flush=True)
                model_full_path = os.path.join(models_dir, model_path)
                model, model_args = get_model(model_full_path)

                random.seed(model_args["seed"])
                torch.manual_seed(model_args["seed"])
                torch.cuda.manual_seed_all(model_args["seed"])
                torch.backends.cudnn.deterministic = True
                np.random.seed(model_args["seed"])

                model_scores = []
                for threshold in thresholds:
                    collate_fn = partial(collate_threshold,
                                        tokenizer=tokenizer,
                                        device=device,
                                        return_attention_masks=False,
                                        pad_to_max_length=False,
                                        threshold=threshold,
                                        collate_orig=coll_call,
                                        n_classes=3)

                    saliency_path_test = os.path.join(test_saliency_dir,
                                                    f'{model_path}_{saliency}')

                    test = NLIDataset(args["dataset_dir"],type="test")
                    test = DatasetSaliency(test, saliency_path_test)

                    test_dl = BucketBatchSampler(batch_size=model_args["batch_size"],
                                                dataset=test,
                                                collate_fn=collate_fn)

                    results = eval_fn(model, test_dl, model_args["labels"])
                    model_scores.append(results[2])

                print(thresholds, model_scores)
                aucs.append(auc(thresholds, model_scores))

            print(f'{np.mean(aucs):.2f} ($\pm${np.std(aucs):.2f})')
