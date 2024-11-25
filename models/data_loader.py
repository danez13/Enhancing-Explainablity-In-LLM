"""Dataset objects and collate functions for all models and datasets."""
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
_NLI_DIC_LABELS = {'entailment': 2, 'neutral': 1, 'contradiction': 0}

def identity(x):
    return x

def collate_threshold(instances: List[Dict],
                      tokenizer: PreTrainedTokenizer,
                      return_attention_masks: bool = True,
                      pad_to_max_length: bool = False,
                      device='cuda',
                      collate_orig=None,
                      threshold=1.0, n_classes=3) -> List[torch.Tensor]:
    batch = collate_orig(instances,
                         tokenizer,
                         return_attention_masks=return_attention_masks,
                         pad_to_max_length=pad_to_max_length,
                         device=device)

    for i, instance in enumerate(batch[0]):
        saliencies = instances[i][-1]
        word_saliencies = [sum([_d[f'{_c}'] for _c in range(n_classes)]) for _d
                           in saliencies]
        sorted_idx = numpy.array(word_saliencies).argsort()[::-1]

        n_tokens = len([_t for _t in instance if _t != tokenizer.pad_token_id])
        num_mask_tokens = int((threshold / 100) * n_tokens)

        num_masked = 0
        if num_mask_tokens > 0:
            for _id in sorted_idx:
                if _id < n_tokens and instance[_id] != tokenizer.pad_token_id:
                    instance[_id] = tokenizer.mask_token_id
                    num_masked += 1
                if num_masked == num_mask_tokens:
                    break

    return batch

def collate_nli(instances: List[Dict],
                tokenizer: PreTrainedTokenizer,
                return_attention_masks: bool = True,
                pad_to_max_length: bool = False,
                device='cuda') -> List[torch.Tensor]:
    token_ids = [tokenizer.encode(_x[0], _x[1], max_length=509,truncation=True) for _x in
                 instances]
    if pad_to_max_length:
        batch_max_len = 512
    else:
        batch_max_len = max([len(_s) for _s in token_ids])

    padded_ids_tensor = torch.tensor(
        [_s + [tokenizer.pad_token_id] * (batch_max_len - len(_s)) for _s in
         token_ids])
    labels = torch.tensor([_x[2] for _x in instances], dtype=torch.long)

    output_tensors = [padded_ids_tensor]
    if return_attention_masks:
        output_tensors.append(padded_ids_tensor > 0)
    output_tensors.append(labels)

    return list(_t.to(device) for _t in output_tensors)

class SortedSampler(Sampler):
    """
    https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/samplers
    /sorted_sampler.html#SortedSampler
    Samples elements sequentially, always in the same order.

    Args:
        data (iterable): Iterable data.
        sort_key (callable): Specifies a function of one argument that is
        used to extract a
            numerical comparison key from each list element.

    Example:
        >>> list(SortedSampler(range(10), sort_key=lambda i: -i))
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    """

    def __init__(self, data, sort_key=identity):
        super().__init__(data)
        self.data = data
        self.sort_key = sort_key
        zip_ = [(i, self.sort_key(row)) for i, row in enumerate(self.data)]
        zip_ = sorted(zip_, key=lambda r: r[1])
        self.sorted_indexes = [item[0] for item in zip_]

    def __iter__(self):
        return iter(self.sorted_indexes)

    def __len__(self):
        return len(self.data)


class BucketBatchSampler(BatchSampler):
    """ https://github.com/PetrochukM/PyTorch-NLP/blob/master/torchnlp
    /samplers/bucket_batch_sampler.py
    `BucketBatchSampler` toggles between `sampler` batches and sorted batches.
    Typically, the `sampler` will be a `RandomSampler` allowing the user to
    toggle between
    random batches and sorted batches. A larger `bucket_size_multiplier` is
    more sorted and vice
    versa.
    Args:
        sampler (torch.data.utils.sampler.Sampler):
        batch_size (int): Size of mini-batch.
        drop_last (bool): If `True` the sampler will drop the last batch if
        its size would be less
            than `batch_size`.
        sort_key (callable, optional): Callable to specify a comparison key
        for sorting.
        bucket_size_multiplier (int, optional): Buckets are of size
            `batch_size * bucket_size_multiplier`.
    Example:
        >>> from torchnlp.random import set_seed
        >>> set_seed(123)
        >>>
        >>> from torch.utils.data.sampler import SequentialSampler
        >>> sampler = SequentialSampler(list(range(10)))
        >>> list(BucketBatchSampler(sampler, batch_size=3, drop_last=False))
        [[6, 7, 8], [0, 1, 2], [3, 4, 5], [9]]
        >>> list(BucketBatchSampler(sampler, batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
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
        if shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        super().__init__(sampler, batch_size, drop_last)
        self.sort_key = sort_key
        self.collate_fn = collate_fn
        self.bucket_sampler = BatchSampler(sampler,
                                           min(
                                               batch_size *
                                               bucket_size_multiplier,
                                               len(sampler)),
                                           False)

    def __iter__(self):
        for bucket in self.bucket_sampler:
            sorted_sampler = SortedSampler([self.dataset[i] for i in bucket],
                                           self.sort_key)
            for batch in SubsetRandomSampler(
                    list(BatchSampler(sorted_sampler, self.batch_size,
                                      self.drop_last))):
                yield self.collate_fn([self.dataset[bucket[i]] for i in batch])

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return math.ceil(len(self.sampler) / self.batch_size)

class DatasetSaliency(Dataset):
    def __init__(self, dataset_cls, sal_dir):
        self._dataset_cls = dataset_cls
        self._dataset = []
        with open(sal_dir) as out:
            for line in out:
                self._dataset.append(json.loads(line)['tokens'])

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, item):
        return tuple(self._dataset_cls[item] + tuple([self._dataset[item]]))

class NLIDataset(Dataset):
    _PATHS = {
        'train': ['esnli_train_1.csv', 'esnli_train_2.csv'],
        'dev': ['esnli_dev.csv'], 'test': ['esnli_test.csv']
    }

    def __init__(self, dir, type='train', sample_dev=False,
                 salient_features=False):
        super().__init__()
        self._dataset = []
        self.salient_features = salient_features
        for _path in [os.path.join(dir, _p) for _p in self._PATHS[type]]:
            with open(_path, encoding="utf-8") as out:
                self._dataset.extend([line for line in
                                  csv.reader(out, delimiter=',')][1:])

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, item):
        result = [self._dataset[item][2], self._dataset[item][3],
                  _NLI_DIC_LABELS[self._dataset[item][1]]]
        if self.salient_features:
            result += [self._dataset[item][5], self._dataset[item][6],
                       self._dataset[item][7], self._dataset[item][8]]
        return tuple(result)


