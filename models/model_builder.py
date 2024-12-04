"""Training utilities."""
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer

# Define the path format for GloVe embeddings with different dimensions (e.g., 50d, 100d, etc.)
_glove_path = "glove.6B.{}d.txt".format


class EarlyStopping:
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                        best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                        best * min_delta / 100)


def _get_glove_embeddings(embedding_dim: int, glove_dir: str):
    """
    Loads GloVe word embeddings from a file and returns a mapping from words to indices 
    and the corresponding word vectors.

    Args:
    - embedding_dim: Dimension of the GloVe embeddings (e.g., 50, 100, 200, 300)
    - glove_dir: Directory where the GloVe embeddings are stored

    Returns:
    - word_to_index: Dictionary mapping words to indices
    - word_vectors: List of word vectors corresponding to words in word_to_index
    """
    word_to_index = {}
    word_vectors = []

    # Open the GloVe file and read it line by line
    with open(os.path.join(glove_dir, _glove_path(embedding_dim)), encoding='utf-8') as fp:
        # Progress bar while loading the embeddings using tqdm
        for line in tqdm(fp.readlines(),
                         desc=f'Loading Glove embeddings from {glove_dir}, '
                         f'dimension {embedding_dim}'):
            line = line.split(" ")  # Split the line by spaces

            word = line[0]  # The first element is the word
            word_to_index[word] = len(word_to_index)  # Assign an index to the word

            # Convert the remaining elements to a numpy array representing the word vector
            vec = np.array([float(x) for x in line[1:]])
            word_vectors.append(vec)

    return word_to_index, word_vectors

def get_embeddings(embedding_dim: int, embedding_dir: str,
                   tokenizer: PreTrainedTokenizer):
    """
    Constructs an embedding matrix based on the tokenizer's vocabulary and GloVe embeddings.

    Args:
    - embedding_dim: Dimension of the embedding (e.g., 50, 100, 200, etc.)
    - embedding_dir: Directory containing the GloVe embeddings
    - tokenizer: Pre-trained tokenizer used to map words to token IDs

    Returns:
    - A tensor with the embedding matrix. The matrix contains word vectors for words 
      in the tokenizer's vocabulary, either loaded from GloVe or randomly initialized.
    """
    # Load GloVe word vectors and corresponding indices
    word_to_index, word_vectors = _get_glove_embeddings(embedding_dim, embedding_dir)

    # Initialize the embedding matrix to zero
    embedding_matrix = np.zeros((len(tokenizer), embedding_dim))

    # For each token ID in the tokenizer's vocabulary, set its word embedding
    for id in range(0, max(tokenizer.vocab.values()) + 1):
        word = tokenizer.ids_to_tokens[id]  # Get the word corresponding to the token ID
        if word not in word_to_index:
            word_vector = np.random.rand(embedding_dim)  # If not found in GloVe, initialize randomly
        else:
            word_vector = word_vectors[word_to_index[word]]  # Use the GloVe vector if available

        embedding_matrix[id] = word_vector  # Assign the embedding vector to the corresponding row

    # Return the embedding matrix as a tensor, which will be a learnable parameter
    return torch.nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float),
                              requires_grad=True)

class CNN_MODEL(torch.nn.Module):
    """
    A Convolutional Neural Network (CNN) model for text classification.
    Uses pre-trained GloVe embeddings, applies convolutional layers over the token embeddings, 
    and uses max-pooling followed by a fully connected layer for classification.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, args: dict, n_labels: int = 2):
        """
        Initializes the CNN model, defining the embedding layer, convolutional layers,
        and the final fully connected layer.

        Args:
        - tokenizer: Pre-trained tokenizer used to map words to tokens
        - args: Dictionary of hyperparameters including the embedding dimension, 
                number of output channels, kernel heights, dropout rate, etc.
        - n_labels: The number of output labels for classification (default is 2 for binary classification)
        """
        super().__init__()
        self.n_labels = n_labels  # Set the number of labels (classes)
        self.args = args  # Save the hyperparameters

        # Embedding layer to convert token IDs to word embeddings
        self.embedding = torch.nn.Embedding(len(tokenizer), args["embedding_dim"])

        # Dropout layer to prevent overfitting
        self.dropout = torch.nn.Dropout(args["dropout"])

        # Set the pre-trained embeddings for the embedding layer
        self.embedding.weight = get_embeddings(args["embedding_dim"], args["embedding_dir"], tokenizer)

        # Define multiple convolutional layers with different kernel heights
        self.conv_layers = torch.nn.ModuleList(
            [torch.nn.Conv2d(args["in_channels"], args["out_channels"], (kernel_height, args["embedding_dim"]), args["stride"], args["padding"]) 
             for kernel_height in args["kernel_heights"]])

        # Fully connected layer that takes the concatenated output from all convolutional layers
        self.final = torch.nn.Linear(len(args["kernel_heights"]) * args["out_channels"], n_labels)

    def conv_block(self, input, conv_layer):
        """
        A function to apply a convolutional block: convolution followed by ReLU activation
        and max pooling.

        Args:
        - input: The input tensor (token embeddings)
        - conv_layer: The convolutional layer to be applied

        Returns:
        - The max-pooled output after applying ReLU activation to the convolutional output
        """
        conv_out = conv_layer(input)  # Apply the convolution
        activation = F.relu(conv_out.squeeze(3))  # Apply ReLU activation
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)  # Max-pooling

        return max_out

    def forward(self, input):
        """
        The forward pass through the CNN model. It includes embedding the input tokens,
        applying dropout, passing the embeddings through the convolutional layers, 
        and finally through the fully connected layer for classification.

        Args:
        - input: The input tensor containing token IDs

        Returns:
        - logits: The output predictions from the final layer (before applying softmax)
        """
        input = self.embedding(input)  # Convert token IDs to word embeddings
        input = input.unsqueeze(1)  # Add an extra dimension for channels (batch_size, 1, seq_len, embedding_dim)
        input = self.dropout(input)  # Apply dropout for regularization

        # Apply each convolutional block and collect the results
        conv_out = [self.conv_block(input, self.conv_layers[i]) for i in range(len(self.conv_layers))]
        # Concatenate the results from all convolutional layers
        all_out = torch.cat(conv_out, 1)
        fc_in = self.dropout(all_out)  # Apply dropout again before the fully connected layer
        logits = self.final(fc_in)  # Pass the concatenated output through the final fully connected layer

        return logits

