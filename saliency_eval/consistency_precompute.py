import json
import os
import random
import traceback
from functools import partial

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer

from models.data_loader import collate_nli, NLIDataset
from models.model_builder import CNN_MODEL


def get_model(model_path, device, tokenizer):
    """
    Loads a model from the specified checkpoint and initializes it with the corresponding parameters.

    Args:
        model_path (str): Path to the model checkpoint.
        device (torch.device): Device to load the model onto (CPU or GPU).
        tokenizer (BertTokenizer): Tokenizer used for text processing.

    Returns:
        tuple: The loaded model and its arguments.
    """
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model_args = checkpoint['args']
    model_cp = CNN_MODEL(tokenizer, model_args, n_labels=checkpoint['args']['labels']).to(device)

    model_cp.load_state_dict(checkpoint['model'])

    return model_cp, model_args


def save_activation(self, inp, out):
    """
    Hook function to save the activations from a specific layer during a forward pass.

    Args:
        self (nn.Module): The layer for which the hook is registered.
        inp (tuple): The input to the layer.
        out (torch.Tensor): The output of the layer.
    """
    global activations
    activations.append(out)


def get_layer_activation(layer, model, instance):
    """
    Computes the activations of a specific layer in the model for a given input instance.

    Args:
        layer (str): Name of the layer to extract activations from.
        model (torch.nn.Module): The model containing the layer.
        instance (tuple): A single data instance from the dataset.

    Returns:
        list: Flattened activations from the specified layer.
    """
    handle = None
    # Register a forward hook on the target layer
    for name, module in model.named_modules():
        if name == layer:
            handle = module.register_forward_hook(save_activation)

    global activations
    activations = []
    with torch.no_grad():  # Disable gradient computation for efficiency
        batch = collate_fn([instance])
        model(batch[0])  # Forward pass with the input batch

    if handle:
        handle.remove()  # Remove the hook after use

    activ1 = None
    try:
        activations = activations[0]
        if isinstance(activations, tuple) and len(activations) == 1:
            activations = activations[0]

        if isinstance(activations[0], torch.nn.utils.rnn.PackedSequence):
            # Handle packed sequences (e.g., for RNNs/LSTMs)
            output, input_sizes = torch.nn.utils.rnn.pad_packed_sequence(
                activations[0], batch_first=True)
            last_idxs = (input_sizes - 1).to(model.device)
            activations = torch.gather(output, 1,
                                       last_idxs.view(-1, 1).unsqueeze(
                                           2).repeat(1, 1,
                                                     model.args.hidden_lstm *
                                                     2)).squeeze()

        # Flatten the activations into a list
        activ1 = activations.detach().cpu().numpy().ravel().tolist()
    except Exception as e:
        print(e, flush=True)
        print(traceback.format_exc(), flush=True)
    return activ1


def get_model_dist(model1, model2, x, layers):
    """
    Computes the difference in activations between two models for a given input across specified layers.

    Args:
        model1 (torch.nn.Module): First model.
        model2 (torch.nn.Module): Second model.
        x (tuple): Input instance.
        layers (list): List of layer names to evaluate.

    Returns:
        list: Differences in activations for each layer.
    """
    dist = []
    for layer in layers:
        act1 = get_layer_activation(layer, model1, x)
        act2 = get_layer_activation(layer, model2, x)
        if not act1 or not act2:
            continue
        # Compute the mean difference between activations
        dist.append(np.mean(np.array(act1).ravel() - np.array(act2).ravel()))
    return dist


def get_layer_names():
    """
    Returns the list of layer names for the CNN model.

    Returns:
        list: Names of layers in the CNN model.
    """
    layers = ["embedding", "conv_layers.0", "conv_layers.1",
              "conv_layers.2", "conv_layers.3", "final"]
    return layers


if __name__ == "__main__":
    # Argument setup and initial configurations
    args = {
        "model_dir_trained": "data/models/snli/cnn/",
        "model_dir_random": "data/models/snli/random_cnn/",
        "output_dir": "data/evaluations/snli/cnn",
        "model": "cnn",
        "dataset_dir": "data/e-SNLI/dataset/",
        "gpu": False,
        "dataset": "snli",
        "model_p": "not",  # Mode for model comparison: trained vs trained, or random vs trained
        "seed": 73  # Seed for reproducibility
    }

    # Seed initialization for reproducibility
    random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])
    torch.backends.cudnn.deterministic = True
    np.random.seed(args["seed"])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load model paths
    models_trained = [_m for _m in os.listdir(args["model_dir_trained"]) if
                      not _m.endswith('.predictions')]
    full_model_paths_trained = [os.path.join(args["model_dir_trained"], _m) for _m
                                in models_trained]

    models_rand = [_m for _m in os.listdir(args["model_dir_random"]) if
                   not _m.endswith('.predictions')]
    full_model_paths_rand = [os.path.join(args['model_dir_random'], _m) for _m in
                             models_rand]

    # Set device (CPU or GPU)
    device = torch.device("cuda") if args["gpu"] else torch.device("cpu")
    test = NLIDataset(args["dataset_dir"], type="test")  # Load test dataset
    coll_call = collate_nli
    pad_to_max_len = False
    collate_fn = partial(coll_call,
                         tokenizer=tokenizer,
                         device=device,
                         return_attention_masks=False,
                         pad_to_max_length=pad_to_max_len)

    # Randomly select two models for comparison
    ind1, ind2 = random.randint(0, 4), random.randint(0, 4)
    print(ind1, ind2, flush=True)

    # Determine which models to compare based on mode
    if args["model_p"] == 'not':
        model1 = full_model_paths_trained[ind1]
        model2 = full_model_paths_trained[ind2]
    elif args["model_p"] == 'rand':
        model1 = full_model_paths_rand[ind1]
        model2 = full_model_paths_rand[ind2]
    else:
        model1 = full_model_paths_rand[ind1]
        model2 = full_model_paths_trained[ind2]

    # Load the selected models
    model1, _ = get_model(model1, device, tokenizer)
    model2, _ = get_model(model2, device, tokenizer)

    diff_activation = []  # To store activation differences
    layers = get_layer_names()  # Get list of layers to evaluate

    # Compute activation differences for all instances in the test set
    for i in tqdm(list(range(0, len(test)))):
        instance = test[i]
        act_dist = get_model_dist(model1, model2, instance, layers)
        diff_activation.append(act_dist)

    # Save the computed activation differences to a file
    with open(f'{args["output_dir"]}/precomp_{args["model"]}_{args["dataset"]}_{args["model_p"]}_{ind1}_{ind2}', 'w') as out:
        out.write(json.dumps(diff_activation) + '\n')

