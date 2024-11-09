"""Training utilities."""
import os
from argparse import Namespace

import numpy as np
import torch
from torch.nn import functional as F, init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
from transformers import PreTrainedTokenizer

_glove_path = "glove.6B.{}d.txt".format


def _get_glove_embeddings(embedding_dim: int, glove_dir: str):
    word_to_index = {}
    word_vectors = []

    with open(os.path.join(glove_dir, _glove_path(embedding_dim)),encoding='utf-8') as fp:
        for line in tqdm(fp.readlines(),
                         desc=f'Loading Glove embeddings from {glove_dir}, '
                         f'dimension {embedding_dim}'):
            line = line.split(" ")

            word = line[0]
            word_to_index[word] = len(word_to_index)

            vec = np.array([float(x) for x in line[1:]])
            word_vectors.append(vec)

    return word_to_index, word_vectors


def get_embeddings(embedding_dim: int, embedding_dir: str,
                   tokenizer: PreTrainedTokenizer):
    """
    :return: a tensor with the embedding matrix - ids of words are from vocab
    """
    word_to_index, word_vectors = _get_glove_embeddings(embedding_dim,
                                                        embedding_dir)

    embedding_matrix = np.zeros((len(tokenizer), embedding_dim))

    for id in range(0, max(tokenizer.vocab.values()) + 1):
        word = tokenizer.ids_to_tokens[id]
        if word not in word_to_index:
            word_vector = np.random.rand(embedding_dim)
        else:
            word_vector = word_vectors[word_to_index[word]]

        embedding_matrix[id] = word_vector

    return torch.nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float),
                              requires_grad=True)

class CNN_MODEL(torch.nn.Module):
    def __init__(self, tokenizer: PreTrainedTokenizer, n_labels: int = 2):
        super().__init__()
        self.n_labels = n_labels
        embedding_dim = 300
        dropout = 0.05
        embedding_dir = "./glove/"
        in_channels = 1
        out_channels = 100
        kernel_heights = [4,5,6,7]
        stride = 1
        padding = 0

        self.embedding = torch.nn.Embedding(len(tokenizer), embedding_dim)

        self.dropout = torch.nn.Dropout(dropout)

        self.embedding.weight = get_embeddings(embedding_dim, embedding_dir, tokenizer)

        self.conv_layers = torch.nn.ModuleList(
            [torch.nn.Conv2d(in_channels, out_channels,(kernel_height, embedding_dim),stride, padding) for kernel_height in kernel_heights])

        self.final = torch.nn.Linear(
            len(kernel_heights) * out_channels, n_labels)

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)
        activation = F.relu(conv_out.squeeze(3))
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)

        return max_out

    def forward(self, input):
        input = self.embedding(input)
        input = input.unsqueeze(1)
        input = self.dropout(input)

        conv_out = [self.conv_block(input, self.conv_layers[i]) for i in
                    range(len(self.conv_layers))]
        all_out = torch.cat(conv_out, 1)
        fc_in = self.dropout(all_out)
        logits = self.final(fc_in)
        return logits
