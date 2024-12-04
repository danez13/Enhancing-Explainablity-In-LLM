"""Script to serialize the saliency scores produced by Shapley Values Sampling"""

# Import necessary libraries
import json
import os
import random
from collections import defaultdict
from functools import partial

import numpy as np
import torch
from captum.attr import ShapleyValueSampling  # For computing Shapley Values saliency
from torch.utils.data import DataLoader  # For data handling in batches
from tqdm import tqdm  # For showing progress bars
from transformers import BertTokenizer  # For tokenizing text using BERT

# Import custom modules for data loading and model building
from models.data_loader import NLIDataset, collate_nli
from models.model_builder import CNN_MODEL
import time

# Function to generate saliency scores using Shapley Value Sampling
def generate_saliency(model_path, saliency_path, args):
    """
    Given a model checkpoint, generate saliency scores using Shapley Value Sampling
    and serialize the results to the specified path.
    """
    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model_args = checkpoint['args']
    model_args["batch_size"] = args["batch_size"] if args["batch_size"] is not None else model_args["batch_size"]

    # Initialize the model and load its state dictionary
    model = CNN_MODEL(tokenizer, model_args, n_labels=checkpoint['args']['labels']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.train()
    model = ModelWrapper(model)  # Wrap the model for Shapley Value Sampling

    # Initialize Shapley Value Sampling object
    ablator = ShapleyValueSampling(model)

    # Set up data collator and data loader for the test dataset
    collate_fn = partial(collate_nli, tokenizer=tokenizer, device=device,
                         return_attention_masks=False, pad_to_max_length=False)

    test = NLIDataset(args["dataset_dir"], type=args["split"], salient_features=True)
    test_dl = DataLoader(batch_size=model_args["batch_size"], dataset=test,
                         shuffle=False, collate_fn=collate_fn)

    # Generate predictions if not already saved
    predictions_path = model_path + '.predictions'
    if not os.path.exists(predictions_path):
        predictions = defaultdict(lambda: [])
        for batch in tqdm(test_dl, desc='Running test prediction... '):
            logits = model(batch[0])
            logits = logits.detach().cpu().numpy().tolist()
            predicted = np.argmax(np.array(logits), axis=-1)
            predictions['class'] += predicted.tolist()
            predictions['logits'] += logits

        # Save predictions to a JSON file
        with open(predictions_path, 'w') as out:
            json.dump(predictions, out)

    # Compute saliency using Shapley Value Sampling
    saliency_flops = []  # List to track the FLOPS (Floating Point Operations) for performance analysis

    # Open the file to write the computed saliency scores
    with open(saliency_path, 'w') as out_mean:
        # Process each batch in the test dataset
        for batch in tqdm(test_dl, desc='Running Saliency Generation...'):
            class_attr_list = defaultdict(lambda: [])  # List to store the attribution values for each class

            additional = None  # Placeholder for any additional arguments (currently unused)

            start = time.time()  # Track the time taken for processing the batch

            token_ids = batch[0].detach().cpu().numpy().tolist()  # Get the token IDs for the current batch

            # Generate Shapley values (saliency scores) for each class
            for cls_ in range(args["labels"]):
                attributions = ablator.attribute(batch[0].float(), target=cls_, additional_forward_args=additional)
                attributions = attributions.detach().cpu().numpy().tolist()
                class_attr_list[cls_] += attributions  # Store the attribution values for this class

            end = time.time()  # Track the end time for batch processing
            saliency_flops.append((end - start) / batch[0].shape[0])  # Calculate FLOPS per token

            # Write the saliency values for each token in the batch to the output file
            for i in range(len(batch[0])):
                saliencies = []  # List to store token saliencies
                for token_i, token_id in enumerate(token_ids[i]):
                    if token_id == tokenizer.pad_token_id:
                        continue  # Skip padding tokens
                    token_sal = {'token': tokenizer.ids_to_tokens[token_id]}  # Get the token corresponding to the ID
                    for cls_ in range(args["labels"]):
                        token_sal[int(cls_)] = class_attr_list[cls_][i][token_i]  # Store saliency for each class
                    saliencies.append(token_sal)

                # Write the saliency information to the file
                out_mean.write(json.dumps({'tokens': saliencies}) + '\n')
                out_mean.flush()  # Ensure data is written immediately

    return saliency_flops  # Return the FLOPS data for analysis

# Wrapper class for the model to enable attribute-based access to the model's forward method
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, input):
        return self.model(input.long())  # Forward pass through the model

# Wrapper class for the BERT model
class BertModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(BertModelWrapper, self).__init__()
        self.model = model

    def forward(self, input):
        return self.model(input.long(), attention_mask=input > 0)[0]  # BERT forward pass with attention mask

# Set up the configuration parameters
args = {
    "dataset_dir": "data/e-SNLI/dataset/",
    "dataset": "snli",
    "split": "test",
    "model": "cnn",
    "gpu": False,
    "seed": 73,
    "labels": 3,
    "models_path": ["data/models/snli/cnn/cnn", "data/models/snli/random_cnn/cnn"],
    "output_dir": ["data/saliency/snli/cnn/", "data/saliency/snli/random_cnn/"],
    "batch_size": None
}

# Set the random seeds for reproducibility
random.seed(args["seed"])
torch.manual_seed(args["seed"])
torch.cuda.manual_seed_all(args["seed"])
torch.backends.cudnn.deterministic = True
np.random.seed(args["seed"])

# Set the device (GPU or CPU)
device = torch.device("cuda") if args["gpu"] else torch.device("cpu")

# Initialize the tokenizer (using BERT)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Loop through models and generate saliency for each one
for model in range(1, 6):
    for models_path, output_dir in zip(args["models_path"], args["output_dir"]):
        model_path = models_path + f"_{model}"
        model_name = model_path.split('/')[-1]

        # Generate and store saliency results
        all_flops = generate_saliency(model_path, os.path.join(output_dir, f'{model_name}_shap'), args)

    # Output the average and standard deviation of FLOPS
    print('FLOPS', np.average(all_flops), np.std(all_flops), flush=True)

