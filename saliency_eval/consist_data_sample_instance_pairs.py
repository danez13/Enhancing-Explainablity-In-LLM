"""
Sampling 4000 instances for evaluating the Data Consistency measure.

This script selects 2000 pairs of instances with the same label and a high overlap in tokens,
and 2000 pairs with different labels and minimal token overlap.
"""

import random

import numpy as np
import torch

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')  # Download the stopwords for English.

from models.data_loader import NLIDataset

# Define stopwords to exclude common words from token comparison.
_stopwords = set(stopwords.words('english'))

if __name__ == "__main__":

    # Configuration for dataset directory and type.
    args = {
        "dataset_dir": "data/e-SNLI/dataset",
        "dataset": "snli",
    }

    # Set a fixed seed for reproducibility.
    seed = 73
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

    # Load the test set from the specified dataset directory.
    test = NLIDataset(args["dataset_dir"], type="test", salient_features=True)

    # Tokenize and filter stopwords for each instance in the test set.
    split_tokens = [set([tok for tok in f'{_t[1]}'.lower().split() if
                         tok not in _stopwords]) for _t in test]
    
    # Extract labels and document IDs from the dataset.
    labels = [_t[2] for _t in test]
    doc_ids = [_i[0].split('.jpg')[0] for _i in test._dataset]

    # Initialize lists to store pairs of instances with the same or different labels.
    same_l = []
    different_l = []

    # Compare all pairs of instances in the test set.
    for i in range(len(test)):
        for j in range(i + 1, len(test)):
            # Check if the labels are the same but document IDs differ.
            if labels[i] == labels[j] and doc_ids[i] != doc_ids[j]:
                # Store pairs with the intersection size of their token sets.
                same_l.append(
                    (i, j, len(split_tokens[i].intersection(split_tokens[j])))
                )
            # Check if the labels are different and document IDs differ.
            if labels[i] != labels[j] and doc_ids[i] != doc_ids[j]:
                different_l.append(
                    (i, j, len(split_tokens[i].intersection(split_tokens[j])))
                )

    # Filter pairs with the same label to only those with at least one token in common.
    same_l = [p for p in same_l if p[-1] >= 1]
    # Sort pairs with the same label by the size of the token intersection in descending order.
    same_l = sorted(same_l, key=lambda x: x[-1], reverse=True)

    # Sort pairs with different labels by the size of the token intersection in ascending order.
    different_l = sorted(different_l, key=lambda x: x[-1])

    # Display details of top pairs from both categories.
    print(same_l[:2])
    print(different_l[:2])
    print(split_tokens[same_l[0][0]])
    print(split_tokens[same_l[0][1]])
    print(split_tokens[different_l[0][0]])
    print(split_tokens[different_l[0][1]])

    # Write selected pairs to a TSV file: 2000 "same label" pairs and 2000 "different label" pairs.
    with open(f'selected_pairs_{args["dataset"]}.tsv', 'w') as out:
        # Take the top 2000 pairs with the same label.
        for inst in same_l[:2000]:
            out.write(f'{inst[0]}\t{inst[1]}\n')
        # Randomly sample 2000 pairs with different labels from the top 5000 pairs.
        for inst in random.sample(different_l[:5000], 2000):
            out.write(f'{inst[0]}\t{inst[1]}\n')

