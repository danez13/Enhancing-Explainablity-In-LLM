"""Script to serialize the saliency scored produced by the Shapley Values
Sampling"""
import argparse
import json
import os
import random
from argparse import Namespace
from collections import defaultdict
from functools import partial

import numpy as np
import torch
from captum.attr import ShapleyValueSampling
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

from models.data_loader import NLIDataset,collate_nli
from models.model_builder import CNN_MODEL


def generate_saliency(model_path, saliency_path,args):
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)
    model_args = checkpoint['args']
    model_args["batch_size"] = args["batch_size"] if args["batch_size"] != None else \
        model_args["batch_size"]

    model = CNN_MODEL(tokenizer, model_args,
                        n_labels=checkpoint['args']['labels']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.train()
    model = ModelWrapper(model)

    ablator = ShapleyValueSampling(model)

    collate_fn = partial(collate_nli, tokenizer=tokenizer, device=device,
                         return_attention_masks=False,
                         pad_to_max_length=False)

    test = NLIDataset(args["dataset_dir"], type=args["split"], salient_features=True)
    test_dl = DataLoader(batch_size=model_args["batch_size"], dataset=test,
                         shuffle=False, collate_fn=collate_fn)

    # PREDICTIONS
    predictions_path = model_path + '.predictions'
    if not os.path.exists(predictions_path):
        predictions = defaultdict(lambda: [])
        for batch in tqdm(test_dl, desc='Running test prediction... '):
            logits = model(batch[0])
            logits = logits.detach().cpu().numpy().tolist()
            predicted = np.argmax(np.array(logits), axis=-1)
            predictions['class'] += predicted.tolist()
            predictions['logits'] += logits

        with open(predictions_path, 'w') as out:
            json.dump(predictions, out)

    # COMPUTE SALIENCY

    saliency_flops = []

    with open(saliency_path, 'w') as out_mean:
        for batch in tqdm(test_dl, desc='Running Saliency Generation...'):
            class_attr_list = defaultdict(lambda: [])

            additional = None

            token_ids = batch[0].detach().cpu().numpy().tolist()

            for cls_ in range(args["labels"]):
                attributions = ablator.attribute(batch[0].float(), target=cls_,
                                                 additional_forward_args=additional)
                attributions = attributions.detach().cpu().numpy().tolist()
                class_attr_list[cls_] += attributions

            for i in range(len(batch[0])):
                saliencies = []
                for token_i, token_id in enumerate(token_ids[i]):
                    if token_id == tokenizer.pad_token_id:
                        continue
                    token_sal = {'token': tokenizer.ids_to_tokens[token_id]}
                    for cls_ in range(["labels"]):
                        token_sal[int(cls_)] = class_attr_list[cls_][i][token_i]
                    saliencies.append(token_sal)

                out_mean.write(json.dumps({'tokens': saliencies}) + '\n')
                out_mean.flush()

    return saliency_flops


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, input):
        return self.model(input.long())


class BertModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(BertModelWrapper, self).__init__()
        self.model = model

    def forward(self, input):
        return self.model(input.long(), attention_mask=input > 0)[0]

args = {
    "dataset_dir":"data/e-SNLI/dataset/",
    "dataset": "snli",
    "split":"test",
    "model":"cnn",
    "gpu":False,
    "seed":73,
    "labels":3,
    "model_path":"data/models/snli/cnn/cnn",
    "output_dir":"data/saliency/snli/cnn/",
    "batch_size":None
}

random.seed(args["seed"])
torch.manual_seed(args["seed"])
torch.cuda.manual_seed_all(args["seed"])
torch.backends.cudnn.deterministic = True
np.random.seed(args["seed"])

device = torch.device("cuda") if args["gpu"] else torch.device("cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

for model in range(1,6):
    model_path = args["models_path"]+f"_{model}"
    model_name = model_path.split('/')[-1]

    all_flops = generate_saliency(model_path, os.path.join(args["output_dir"],
                                                            f'{model_name}_shap'),
                                                            args)

print('FLOPS', np.average(all_flops), np.std(all_flops), flush=True)
