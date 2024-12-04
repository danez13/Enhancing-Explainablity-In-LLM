"""
Evaluating Consistency Rationale measure.

This script evaluates the consistency between model activations and saliency measures
across different layers of a CNN model. It computes Spearman rank correlation between 
normalized differences in activations and saliency scores.
"""

# Import necessary libraries
import json
import os
import random
from functools import partial

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
from transformers import BertTokenizer

# Import custom modules for data loading and model building
from models.data_loader import collate_nli, NLIDataset
from models.model_builder import CNN_MODEL

def get_model(model_path, device, model_type, tokenizer):
    """
    Load the model from a checkpoint file.

    Args:
        model_path (str): Path to the model checkpoint.
        device (torch.device): Device to load the model (CPU or GPU).
        model_type (str): Type of the model (e.g., CNN).
        tokenizer (BertTokenizer): Tokenizer for preprocessing text.

    Returns:
        tuple: The loaded model and its arguments.
    """
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model_args = checkpoint['args']
    model_cp = CNN_MODEL(tokenizer, model_args, n_labels=checkpoint['args']['labels']).to(device)
    model_cp.load_state_dict(checkpoint['model'])

    return model_cp, model_args

def get_saliencies(saliency_path):
    """
    Load saliency maps from a file.

    Args:
        saliency_path (str): Path to the saliency file.

    Returns:
        list: List of token-level saliency scores for each instance.
    """
    result = []
    n_labels = 3  # Number of classification labels
    with open(saliency_path) as out:
        for line in out:
            instance_saliency = json.loads(line)
            saliency = instance_saliency['tokens']
            token_pred_saliency = []
            for _cls in range(0, n_labels):
                for record in saliency:
                    token_pred_saliency.append(record[str(_cls)])
            result.append(token_pred_saliency)
    return result

def get_layer_names(model, dataset):
    """
    Return the names of the layers in the CNN model.

    Args:
        model (str): Model name.
        dataset (str): Dataset name.

    Returns:
        list: Layer names.
    """
    layers = ["embedding", "conv_layers.0", "conv_layers.1",
              "conv_layers.2", "conv_layers.3", "final"]
    return layers

def get_sal_dist(sal1, sal2):
    """
    Compute the mean absolute difference between two saliency maps.

    Args:
        sal1 (list): Saliency scores from the first model.
        sal2 (list): Saliency scores from the second model.

    Returns:
        float: Mean absolute difference.
    """
    return np.mean(np.abs(np.array(sal1).reshape(-1) - np.array(sal2).reshape(-1)))

if __name__ == "__main__":
    # Configuration settings
    args = {
        "model_dir_trained": "data/models/snli/cnn/",
        "model_dir_random": "data/models/snli/random_cnn/",
        "output_dir": "data/evaluations/snli/cnn",
        "saliency_dir_trained": "data/saliency/snli/cnn",
        "saliency_dir_random": "data/saliency/snli/random_cnn",
        "saliencies": ["rand", "shap", "sal_mean", "sal_l2", "occlusion_none",
                       "lime", "inputx_mean", "inputx_l2", "guided_mean", "guided_l2"],
        "model": "cnn",
        "gpu": False,
        "dataset": "snli",
        "dataset_dir": "data/e-SNLI/dataset",
        "per_layer": False,  # Whether to evaluate per-layer consistency
    }
    print(args, flush=True)

    # Set random seeds for reproducibility
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    np.random.seed(1)

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    for saliency in args["saliencies"]:
        # Load trained and random models
        models_trained = [m for m in os.listdir(args["model_dir_trained"]) if not m.endswith('.predictions')]
        saliency_trained = [os.path.join(args["saliency_dir_trained"], m + f'_{saliency}') for m in models_trained]

        models_rand = [m for m in os.listdir(args["model_dir_random"]) if not m.endswith('.predictions')]
        saliency_rand = [os.path.join(args["saliency_dir_random"], m + f'_{saliency}') for m in models_rand]

        device = torch.device("cuda") if args["gpu"] else torch.device("cpu")
        
        # Load test dataset
        test = NLIDataset(args["dataset_dir"], type="test", salient_features=True)
        coll_call = collate_nli
        collate_fn = partial(coll_call, tokenizer=tokenizer, device=device,
                             return_attention_masks=False, pad_to_max_length=False,
                             collate_orig=coll_call, n_classes=3)

        # Precomputed activation differences
        precomputed = []
        for f in os.scandir(args["output_dir"]):
            if args["model"] in f.name and args["dataset"] in f.name and f.name.startswith('precomp_'):
                precomputed.append(f.name)

        diff_activation, diff_saliency = [], []

        # Compare activations and saliencies
        for f in precomputed:
            act_distances = json.load(open(f'{args["output_dir"]}/{f}'))
            ids = [int(n) for n in f.split('_') if n.isdigit()]
            model_p = f.split('_')[3]
            
            if model_p == 'not':
                saliencies1 = get_saliencies(saliency_trained[ids[0]])
                saliencies2 = get_saliencies(saliency_trained[ids[1]])
            elif model_p == 'rand':
                saliencies1 = get_saliencies(saliency_rand[ids[0]])
                saliencies2 = get_saliencies(saliency_rand[ids[1]])
            else:
                saliencies1 = get_saliencies(saliency_rand[ids[0]])
                saliencies2 = get_saliencies(saliency_trained[ids[1]])

            # Process each instance
            for inst_id in range(len(test)):
                try:
                    sal_dist = get_sal_dist(saliencies1[inst_id], saliencies2[inst_id])
                    act_dist = act_distances[inst_id]
                    diff_activation.append(act_dist)
                    diff_saliency.append(sal_dist)
                except:
                    continue

        # Evaluate per-layer or aggregated consistency
        if args["per_layer"]:
            for i in range(len(diff_activation[0])):
                acts = [np.abs(dist[i]) for dist in diff_activation]
                diff_act = MinMaxScaler().fit_transform(np.array(acts).reshape(-1, 1)).flatten()
                diff_sal = MinMaxScaler().fit_transform(np.array(diff_saliency).reshape(-1, 1)).flatten()
                sp = spearmanr(diff_act, diff_sal)
                print(f'{sp[0]:.3f}', flush=True, end=' ')
        else:
            acts = [np.abs(np.mean(dist)) for dist in diff_activation]
            diff_act = MinMaxScaler().fit_transform(np.array(acts).reshape(-1, 1)).flatten()
            diff_sal = MinMaxScaler().fit_transform(np.array(diff_saliency).reshape(-1, 1)).flatten()
            sp = spearmanr(diff_act, diff_sal)
            print(f'\n{sp[0]:.3f} ({sp[1]:.1e})', flush=True)

        # Save results
        output_file = f"{args['output_dir']}/cnn_consistency_{saliency}"
        with open(output_file, "w") as file:
            file.write(f"{sp[0]:.3f} {sp[1]:.1e}\n")

