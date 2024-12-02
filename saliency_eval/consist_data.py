"""Evaluates saliencies with Data Consistency measure."""
import argparse
import json
import os
import traceback
from functools import partial

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from transformers import BertTokenizer

from models.data_loader import collate_nli, NLIDataset
from saliency_eval.consistency_rats import get_layer_names, get_model


def get_saliencies(saliency_path):
    result = []
    tokens = []
    print(saliency_path, flush=True)
    with open(saliency_path) as out:
        for i, line in enumerate(out):
            if i >= len(test):
                break
            instance_saliency = json.loads(line)
            saliency = instance_saliency['tokens']

            instance = test[i]

            token_ids = tokenizer.encode(instance[0], instance[1])

            token_pred_saliency = []

            n_labels = 3
            for _cls in range(0, n_labels):
                for record in saliency:
                    token_pred_saliency.append(record[str(_cls)])

            result.append(token_pred_saliency)
            tokens.append(token_ids)
    return result, tokens


def save_activation(self, inp, out):
    global activations
    activations.append(out)


def get_layer_activation(layer, model, instance):
    handle = None
    for name, module in model.named_modules():
        # partial to assign the layer name to each hook
        if name == layer:
            handle = module.register_forward_hook(save_activation)

    global activations
    activations = []
    with torch.no_grad():
        batch = collate_fn([instance])
        model(batch[0])

    if handle:
        handle.remove()

    activ1 = None
    try:
        activations = activations[0]
        if isinstance(activations, tuple) and len(activations) == 1:
            activations = activations[0]

        if isinstance(activations[0], torch.nn.utils.rnn.PackedSequence):
            output, input_sizes = torch.nn.utils.rnn.pad_packed_sequence(
                activations[0], batch_first=True)
            last_idxs = (input_sizes - 1).to(model.device)
            activations = torch.gather(output, 1,
                                       last_idxs.view(-1, 1).unsqueeze(
                                           2).repeat(1, 1,
                                                     model.args.hidden_lstm *
                                                     2)).squeeze()

        activ1 = activations.detach().cpu().numpy().ravel().tolist()
    except Exception as e:
        print(e, flush=True)
        print(print(traceback.format_exc()), flush=True)
    return activ1


def get_model_distv2(model, x, y,args):
    dist = []
    layer_names = get_layer_names(args["model"], args["dataset"])
    for layer in layer_names:
        act1 = get_layer_activation(layer, model, x)
        act2 = get_layer_activation(layer, model, y)
        if act2 is None or act1 is None:
            continue
        dist.append(np.mean(np.array(act1).ravel() - np.array(act2).ravel()))
    return dist


def get_model_embedding_emb_size(model,):
    return model.embedding.weight.shape[0]


if __name__ == "__main__":
    args = {
        "model_dir_trained": "data/models/snli/cnn/",
        "model_dir_random": "data/models/snli/random_cnn/",
        "saliency_dir_random": "data/saliency/snli/random_cnn/",
        "saliency_dir_trained": "data/saliency/snli/cnn/",
        "saliencies": ["rand","shap", "sal_mean", "sal_l2", "occlusion_none", "lime", "inputx_mean", "inputx_l2", "guided_mean", "guided_l2"],
        "model": "cnn",
        "dataset_dir": "data/e-SNLI/dataset",
        "gpu": False,
        "dataset": "snli",
        "output_dir": "data/evaluations/snli/cnn"
    }

    np.random.seed(1)

    for saliency in args["saliencies"]:
        print(saliency)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        models_trained = [_m for _m in os.listdir(args["model_dir_trained"])
                          if not _m.endswith('.predictions')]
        full_model_paths_trained = [os.path.join(args["model_dir_trained"], _m) for
                                    _m in models_trained]
        saliency_trained = [
            os.path.join(args["saliency_dir_trained"], _m + f'_{saliency}') for _m
            in models_trained]

        device = torch.device("cuda") if args["gpu"] else torch.device("cpu")
        test = NLIDataset(args["dataset_dir"],type="test")
        return_attention_masks = False

        coll_call = collate_nli
        collate_fn = partial(coll_call, tokenizer=tokenizer, device=device,
                             return_attention_masks=True,
                             pad_to_max_length=True)

        dataset_ids = []
        with open(f'selected_pairs_{args["dataset"]}.tsv') as out:
            for line in out:
                line = line.strip().split()
                dataset_ids.append((int(line[0]), int(line[1])))

        all_scores = []
        for i, model_path in enumerate(full_model_paths_trained):
            print(model_path, flush=True)
            model, model_args = get_model(model_path, device, args["model"],
                                          tokenizer)
            model_size = get_model_embedding_emb_size(model)

            saliencies, tokens = get_saliencies(saliency_trained[i])

            diff_activation, diff_saliency = [], []
            dist_dir = f'consist_data/' \
                f'{args["dataset"]}_{model_path.split("/")[-1]}'
            if not os.path.exists(dist_dir):
                for i, (ind1, ind2) in tqdm(enumerate(dataset_ids),
                                            desc='Loading Model Differences'):
                    model_dist = get_model_distv2(model, test[int(ind1)],
                                                  test[int(ind2)],
                                                  args)
                    diff_activation.append(model_dist)

                with open(dist_dir, 'w') as out:
                    json.dump(diff_activation, out)
            else:
                diff_activation = json.load(open(dist_dir))
            for i, (ind1, ind2) in tqdm(enumerate(dataset_ids),
                                        desc='Loading Sal Differences'):
                if ind1 >= len(saliencies) or ind2 >= len(saliencies):
                    continue
                pair_word_mask = [0.0] * model_size
                mult1 = [0.0000001] * model_size
                for token_id, sal in zip(tokens[ind1], saliencies[ind1]):
                    mult1[token_id] = sal
                    pair_word_mask[token_id] = 1.0

                mult2 = [0.0000001] * model_size
                for token_id, sal in zip(tokens[ind2], saliencies[ind2]):
                    mult2[token_id] = sal
                    pair_word_mask[token_id] = 1.0
                mult1 = np.array(
                    [v for i, v in enumerate(mult1) if pair_word_mask[i] != 0])
                mult2 = np.array(
                    [v for i, v in enumerate(mult2) if pair_word_mask[i] != 0])

                sal_dist = np.mean(np.abs(mult1 - mult2))
                diff_saliency.append(sal_dist)

            print(diff_saliency[:10])
            diff_activation = MinMaxScaler().fit_transform(
                [[np.mean(np.abs(_d))] for _d in diff_activation])
            diff_saliency = MinMaxScaler().fit_transform(
                [[_d] for _d in diff_saliency])
            diff_activation = [_d[0] for _d in diff_activation]
            diff_saliency = [_d[0] for _d in diff_saliency]

            diff_activation = np.nan_to_num(diff_activation)
            diff_saliency = np.nan_to_num(diff_saliency)
            if len(diff_activation) != len(diff_saliency):
                continue
            sr = spearmanr(diff_activation, diff_saliency)
            all_scores.append([sr[0], sr[1]])
            print(sr, flush=True)

        print(f'\n{np.mean([_scores[0] for _scores in all_scores]):.3f} '
              f'({np.mean([_scores[1] for _scores in all_scores]):.1e})\n',
              flush=True)
        with open(args["output_dir"],"w") as file:
            file.write(f"{np.mean([_scores[0] for _scores in all_scores]):.3f} {np.mean([_scores[1] for _scores in all_scores]):.1e}")