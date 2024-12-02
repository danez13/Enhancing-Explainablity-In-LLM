"""Sampling of 4000 instances to be used for evaluation of Data Consistency
measure."""
import argparse
import random

import numpy as np
import torch
from nltk.corpus import stopwords

from models.data_loader import NLIDataset

_stopwords = set(stopwords.words('english'))

if __name__ == "__main__":

    args = {
        "dataset_dir": "data/e-SNLI/dataset",
        "dataset": "snli",
    }
    seed = 73
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

    test = NLIDataset(args["dataset_dir"], type="test", salient_features=True)

    split_tokens = [set([tok for tok in f'{_t[1]}'.lower().split() if
                            tok not in _stopwords]) for _t in test]
    labels = [_t[2] for _t in test]
    doc_ids = [_i[0].split('.jpg')[0] for _i in test._dataset]

    same_l = []
    different_l = []

    for i in range(len(test)):
        for j in range(i + 1, len(test)):
            if labels[i] == labels[j] and doc_ids[i] != doc_ids[j]:
                same_l.append(
                    (i, j, len(split_tokens[i].intersection(split_tokens[j]))))
            if labels[i] != labels[j] and doc_ids[i] != doc_ids[j]:
                different_l.append(
                    (i, j, len(split_tokens[i].intersection(split_tokens[j]))))

    same_l = [p for p in same_l if p[-1] >= 1]
    same_l = sorted(same_l, key=lambda x: x[-1], reverse=True)

    different_l = sorted(different_l, key=lambda x: x[-1])

    print(same_l[:2])
    print(different_l[:2])
    print(split_tokens[same_l[0][0]])
    print(split_tokens[same_l[0][1]])
    print(split_tokens[different_l[0][0]])
    print(split_tokens[different_l[0][1]])

    with open(f'selected_pairs_{args["dataset"]}.tsv', 'w') as out:
        for inst in same_l[:2000]:
            out.write(f'{inst[0]}\t{inst[1]}\n')
        for inst in random.sample(different_l[:5000], 2000):
            out.write(f'{inst[0]}\t{inst[1]}\n')
