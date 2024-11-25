"""Evaluating Consistency Rationale measure."""
import argparse
import json
import os
import random
import traceback
from argparse import Namespace
from functools import partial

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
from transformers import BertConfig, BertForSequenceClassification, \
    BertTokenizer

from models.data_loader import collate_nli, NLIDataset
from models.model_builder import CNN_MODEL

def get_saliencies(saliency_path):
    result = []
    n_labels = 3
    with open(saliency_path) as out:
        for i, line in enumerate(out):
            instance_saliency = json.loads(line)
            saliency = instance_saliency['tokens']
            token_pred_saliency = []
            for _cls in range(0, n_labels):
                for record in saliency:
                    token_pred_saliency.append(record[str(_cls)])

            result.append(token_pred_saliency)
    return result

def get_layer_names(model, dataset):
    layers = ["embedding", "conv_layers.0", "conv_layers.1",
                      "conv_layers.2", "conv_layers.3", "final"]
    return layers


def get_sal_dist(sal1, sal2):
    return np.mean(
        np.abs(np.array(sal1).reshape(-1) - np.array(sal2).reshape(-1)))


if __name__ == "__main__":
    args = {
        "model_dir_trained": "data/models/snli/cnn/",
        "model_dir_random": "data/models/snli/random_cnn/",
        "output_dir": "data/evaluations/snli/cnn",
        "saliency_dir_trained":"data/saliency/snli/cnn",
        "saliency_dir_random":"data/saliency/snli/random_cnn",
        "saliencies": ["rand","shap", "sal_mean", "sal_l2", "occlusion_none", "lime", "inputx_mean", "inputx_l2", "guided_mean", "guided_l2"],
        "model": "cnn",
        "gpu": False,
        "dataset":"snli",
        "dataset_dir": "data/e-SNLI/dataset",
        "per_layer": False,
    }
    print(args, flush=True)

    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    np.random.seed(1)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for saliency in args["saliencies"]:
        models_trained = [_m for _m in os.listdir(args["model_dir_trained"]) if
                          not _m.endswith('.predictions')]
        saliency_trained = [
            os.path.join(args["saliency_dir_trained"], _m + f'_{saliency}') for _m
            in models_trained]

        models_rand = [_m for _m in os.listdir(args["model_dir_random"]) if
                       not _m.endswith('.predictions')]
        saliency_rand = [
            os.path.join(args["saliency_dir_random"], _m + f'_{saliency}') for _m
            in models_rand]

        return_attention_masks = args.model == 'trans'

        device = torch.device("cuda") if args["gpu"] else torch.device("cpu")
        test = NLIDataset(args["dataset_dir"], type="test", salient_features=True)
        coll_call = collate_nli
        collate_fn = partial(coll_call,
                             tokenizer=tokenizer,
                             device=device,
                             return_attention_masks=return_attention_masks,
                             pad_to_max_length=False,
                             collate_orig=coll_call,
                             n_classes=3 if args["dataset"] in ['snli',
                                                             'tweet'] else 2)

        layers = get_layer_names(args["model"], args["dataset"])

        precomputed = []
        for _f in os.scandir('consist_rat'):
            _f = _f.name
            if args["model"] in _f and args["dataset"] in _f and _f.startswith(
                    'precomp_'):
                precomputed.append(_f)

        diff_activation, diff_saliency = [], []
        for f in precomputed:
            act_distances = json.load(open('consist_rat/' + f))
            ids = [int(_n) for _n in f.split('_') if _n.isdigit()]
            model_p = f.split('_')[3]
            if model_p == 'not':
                saliencies1, saliencies2 = get_saliencies(
                    saliency_trained[ids[0]]), get_saliencies(
                    saliency_trained[ids[1]])
            elif model_p == 'rand':
                saliencies1, saliencies2 = get_saliencies(
                    saliency_rand[ids[0]]), get_saliencies(
                    saliency_rand[ids[1]])
            else:
                saliencies1, saliencies2 = get_saliencies(
                    saliency_rand[ids[0]]), get_saliencies(
                    saliency_trained[ids[1]])

            for inst_id in range(len(test)):
                try:
                    sal_dist = get_sal_dist(saliencies1[inst_id],
                                            saliencies2[inst_id])
                    act_dist = act_distances[inst_id]
                    diff_activation.append(act_dist)
                    diff_saliency.append(sal_dist)
                except:
                    continue

        if args["per_layer"]:
            for i in range(len(diff_activation[0])):
                acts = [np.abs(_dist[i]) for _dist in diff_activation]
                diff_act = MinMaxScaler().fit_transform([[_d] for _d in acts])
                diff_act = [_d[0] for _d in diff_act]

                diff_sal = MinMaxScaler().fit_transform(
                    [[_d] for _d in diff_saliency])
                diff_sal = [_d[0] for _d in diff_sal]

                sp = spearmanr(diff_act, diff_sal)
                print(f'{sp[0]:.3f}', flush=True, end=' ')

        acts = [np.abs(np.mean(_dist)) for _dist in diff_activation]
        diff_act = MinMaxScaler().fit_transform([[_d] for _d in acts])
        diff_act = [_d[0] for _d in diff_act]

        diff_sal = MinMaxScaler().fit_transform([[_d] for _d in diff_saliency])
        diff_sal = [_d[0] for _d in diff_sal]

        sp = spearmanr(diff_act, diff_sal)
        print()
        print(f'{sp[0]:.3f} ({sp[1]:.1e})', flush=True)
