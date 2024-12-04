"""Dataset objects and collate functions for all models and datasets."""
# Import necessary libraries
import csv
import json
import math
import os
from typing import Dict, List

import numpy
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler, Sampler, \
    SequentialSampler, SubsetRandomSampler
from transformers import PreTrainedTokenizer

# Mapping for NLI labels (entailment, neutral, contradiction)
_NLI_DIC_LABELS = {'entailment': 2, 'neutral': 1, 'contradiction': 0}

# Helper function for identity (used in sorting operations)
def identity(x):
    return x

# Collate function that applies a threshold to mask tokens based on saliency scores
def collate_threshold(instances: List[Dict],
                      tokenizer: PreTrainedTokenizer,
                      return_attention_masks: bool = True,
                      pad_to_max_length: bool = False,
                      device='cuda',
                      collate_orig=None,
                      threshold=1.0, n_classes=3) -> List[torch.Tensor]:
    """
    Applies a saliency-based threshold to mask tokens. The function first uses the original collate
    function to prepare the batch and then masks the tokens with the highest saliency scores.
    """
    # Collate data using the original collate function
    batch = collate_orig(instances,
                         tokenizer,
                         return_attention_masks=return_attention_masks,
                         pad_to_max_length=pad_to_max_length,
                         device=device)

    # Iterate over each instance in the batch
    for i, instance in enumerate(batch[0]):
        # Get the saliency scores for the current instance
        saliencies = instances[i][-1]
        # Aggregate saliency values by class
        word_saliencies = [sum([_d[f'{_c}'] for _c in range(n_classes)]) for _d in saliencies]
        # Sort the indices of words based on their saliency values in descending order
        sorted_idx = numpy.array(word_saliencies).argsort()[::-1]

        # Calculate the number of tokens to mask based on the threshold
        n_tokens = len([_t for _t in instance if _t != tokenizer.pad_token_id])
        num_mask_tokens = int((threshold / 100) * n_tokens)

        num_masked = 0
        # Mask the top tokens with the highest saliency scores
        if num_mask_tokens > 0:
            for _id in sorted_idx:
                if _id < n_tokens and instance[_id] != tokenizer.pad_token_id:
                    instance[_id] = tokenizer.mask_token_id
                    num_masked += 1
                if num_masked == num_mask_tokens:
                    break

    # Return the modified batch
    return batch

# Collate function for standard NLI (without saliency-based modifications)
def collate_nli(instances: List[Dict],
                tokenizer: PreTrainedTokenizer,
                return_attention_masks: bool = True,
                pad_to_max_length: bool = False,
                device='cuda') -> List[torch.Tensor]:
    """
    Standard collate function for tokenizing and batching NLI instances.
    It handles tokenization, padding, and creating attention masks.
    """
    # Tokenize each premise-hypothesis pair into token IDs
    token_ids = [tokenizer.encode(_x[0], _x[1], max_length=509, truncation=True) for _x in instances]
    
    # Determine the maximum length for padding (either fixed or dynamic)
    if pad_to_max_length:
        batch_max_len = 512
    else:
        batch_max_len = max([len(_s) for _s in token_ids])

    # Pad tokenized inputs to the same length
    padded_ids_tensor = torch.tensor(
        [_s + [tokenizer.pad_token_id] * (batch_max_len - len(_s)) for _s in token_ids])
    
    # Convert the labels to tensors
    labels = torch.tensor([_x[2] for _x in instances], dtype=torch.long)

    # Create output tensors for token IDs, attention mask, and labels
    output_tensors = [padded_ids_tensor]
    if return_attention_masks:
        output_tensors.append(padded_ids_tensor > 0)  # Attention mask (1 for real tokens, 0 for padding)
    output_tensors.append(labels)

    # Move tensors to the specified device (CPU or GPU)
    return list(_t.to(device) for _t in output_tensors)

# Custom sampler that sorts data by a given key
class SortedSampler(Sampler):
    """
    Custom sampler that sorts data by a given key and returns the sorted indices for iteration.
    It ensures that the samples are retrieved in a sorted order based on the provided sorting key.
    """
    def __init__(self, data, sort_key=identity):
        super().__init__(data)
        self.data = data
        self.sort_key = sort_key
        # Sort data by the specified key and store the sorted indices
        zip_ = [(i, self.sort_key(row)) for i, row in enumerate(self.data)]
        zip_ = sorted(zip_, key=lambda r: r[1])
        self.sorted_indexes = [item[0] for item in zip_]

    def __iter__(self):
        # Yield the sorted indices for iteration
        return iter(self.sorted_indexes)

    def __len__(self):
        # Return the number of samples
        return len(self.data)

# Custom batch sampler that batches data using a combination of random and sorted sampling
class BucketBatchSampler(BatchSampler):
    """
    Custom batch sampler that combines random sampling with sorted batching.
    It organizes data into buckets based on sequence length to improve batch efficiency.
    """
    def __init__(self,
                 dataset: Dataset,
                 batch_size,
                 collate_fn,
                 drop_last=False,
                 shuffle=True,
                 sort_key=identity,
                 bucket_size_multiplier=100):
        self.dataset = dataset
        # Initialize the sampler (random or sequential)
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        super().__init__(sampler, batch_size, drop_last)
        self.sort_key = sort_key
        self.collate_fn = collate_fn
        # Create a batch sampler that groups data into buckets
        self.bucket_sampler = BatchSampler(sampler,
                                           min(batch_size * bucket_size_multiplier, len(sampler)),
                                           False)

    def __iter__(self):
        # For each bucket, sample data in a sorted order and create batches
        for bucket in self.bucket_sampler:
            sorted_sampler = SortedSampler([self.dataset[i] for i in bucket], self.sort_key)
            for batch in SubsetRandomSampler(list(BatchSampler(sorted_sampler, self.batch_size, self.drop_last))):
                yield self.collate_fn([self.dataset[bucket[i]] for i in batch])

    def __len__(self):
        # Return the number of batches
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return math.ceil(len(self.sampler) / self.batch_size)

# Custom dataset class that integrates saliency features with the original dataset
class DatasetSaliency(Dataset):
    def __init__(self, dataset_cls, sal_dir):
        """
        This dataset class extends the original dataset by loading saliency data from a file
        and appending it to each instance.
        """
        self._dataset_cls = dataset_cls
        self._dataset = []
        # Load saliency data from the specified directory
        with open(sal_dir) as out:
            for line in out:
                self._dataset.append(json.loads(line)['tokens'])

    def __len__(self):
        # Return the length of the dataset
        return len(self._dataset)

    def __getitem__(self, item):
        # Return the original dataset instance along with its corresponding saliency features
        return tuple(self._dataset_cls[item] + tuple([self._dataset[item]]))

# Custom NLI dataset class for loading and processing e-SNLI data
class NLIDataset(Dataset):
    _PATHS = {
        'train': ['esnli_train_1.csv', 'esnli_train_2.csv'],
        'dev': ['esnli_dev.csv'],
        'test': ['esnli_test.csv']
    }

    def __init__(self, dir, type='train', sample_dev=False, salient_features=False):
        """
        This dataset class loads the e-SNLI dataset from CSV files and processes it for use in training.
        """
        super().__init__()
        self._dataset = []
        self.salient_features = salient_features
        # Load data from CSV files based on the dataset type (train, dev, test)
        for _path in [os.path.join(dir, _p) for _p in self._PATHS[type]]:
            with open(_path, encoding="utf-8") as out:
                self._dataset.extend([line for line in csv.reader(out, delimiter=',')][1:])

    def __len__(self):
        # Return the length of the dataset
        return len(self._dataset)

    def __getitem__(self, item):
        # Return a tuple containing premise, hypothesis, and label, with optional salient features
        result = [self._dataset[item][2], self._dataset[item][3],
                  _NLI_DIC_LABELS[self._dataset[item][1]]]
        if self.salient_features:
            result += [self._dataset[item][5], self._dataset[item][6],
                       self._dataset[item][7], self._dataset[item][8]]
        return tuple(result)