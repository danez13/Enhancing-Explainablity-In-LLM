"""Script to serialize the saliencies from the LIME method."""

# Import necessary libraries
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from lime.lime_text import LimeTextExplainer  # For LIME explanation
from tqdm import tqdm  # For progress bars
from transformers import BertTokenizer  # For tokenizing text using BERT

# Import custom modules for data loading and model building
from models.data_loader import NLIDataset
from models.model_builder import CNN_MODEL
import time

# Wrapper class to handle BERT model for LIME compatibility
class BertModelWrapper(nn.Module):
    def __init__(self, model, device, tokenizer, args):
        super(BertModelWrapper, self).__init__()
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

    def forward(self, token_ids):
        results = []
        # Preprocess token IDs (convert them into integer format)
        token_ids = [[int(i) for i in instance_ids.split(' ') if i != ''] for instance_ids in token_ids]
        for i in tqdm(range(0, len(token_ids), self.args["batch_size"]), 'Building a local approximation...'):
            # Process tokens in batches
            batch_ids = token_ids[i:i + self.args["batch_size"]]
            max_batch_id = min(max([len(_l) for _l in batch_ids]), 512)  # Limit sequence length to 512
            batch_ids = [_l[:max_batch_id] for _l in batch_ids]  # Truncate sequences
            padded_batch_ids = [
                _l + [self.tokenizer.pad_token_id] * (max_batch_id - len(_l)) for _l in batch_ids
            ]  # Pad the sequences
            tokens_tensor = torch.tensor(padded_batch_ids).to(self.device)  # Convert to tensor
            logits = self.model(tokens_tensor.long(), attention_mask=tokens_tensor.long() > 0)  # Forward pass
            results += logits[0].detach().cpu().numpy().tolist()  # Store logits (output)
        return np.array(results)

# Wrapper class for CNN model for LIME explanation
class ModelWrapper(nn.Module):
    def __init__(self, model, device, tokenizer, args):
        super(ModelWrapper, self).__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.args = args

    def forward(self, token_ids):
        results = []
        # Preprocess token IDs (convert them into integer format)
        token_ids = [[int(i) for i in instance_ids.split(' ') if i != ''] for instance_ids in token_ids]
        for i in tqdm(range(0, len(token_ids), self.args["batch_size"]), 'Building a local approximation...'):
            # Process tokens in batches
            batch_ids = token_ids[i:i + self.args["batch_size"]]
            max_batch_id = max([len(_l) for _l in batch_ids])  # Get the maximum sequence length
            padded_batch_ids = [
                _l + [self.tokenizer.pad_token_id] * (max_batch_id - len(_l)) for _l in batch_ids
            ]  # Pad sequences to the maximum length in the batch
            tokens_tensor = torch.tensor(padded_batch_ids).to(self.device)  # Convert to tensor
            logits = self.model(tokens_tensor)  # Forward pass through the model
            results += logits.detach().cpu().numpy().tolist()  # Store the logits (model output)
        return np.array(results)

# Function to generate saliency scores using the LIME method
def generate_saliency(model_path, saliency_path, args):
    """
    Given a model checkpoint, generate saliency scores using the LIME method
    and serialize the results to the specified path.
    """
    test = NLIDataset(args["dataset_dir"], type=args["split"], salient_features=True)  # Load test data
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)  # Load model checkpoint
    model_args = checkpoint['args']  # Extract model arguments

    model_args["batch_size"] = 300  # Set batch size
    # Initialize model and load weights
    model = CNN_MODEL(tokenizer, model_args, n_labels=checkpoint['args']['labels']).to(device)
    model.load_state_dict(checkpoint['model'])
    modelw = ModelWrapper(model, device, tokenizer, model_args)  # Wrap the model

    modelw.eval()  # Set the model to evaluation mode

    explainer = LimeTextExplainer()  # Initialize LIME explainer
    saliency_flops = []  # List to store FLOPS (computational time per sample)

    with open(saliency_path, 'w') as out:
        # Process each instance in the dataset
        for instance in tqdm(test):
            start = time.time()  # Track the start time for processing
            saliencies = []
            token_ids = tokenizer.encode(instance[0], instance[1])  # Tokenize input text

            # Pad token IDs if the length is less than 6
            if len(token_ids) < 6:
                token_ids = token_ids + [tokenizer.pad_token_id] * (6 - len(token_ids))
            
            try:
                # Generate LIME explanation for the instance
                exp = explainer.explain_instance(
                    " ".join([str(i) for i in token_ids]), modelw,
                    num_features=len(token_ids), top_labels=args["labels"]
                )
            except Exception as e:
                print(e)

                # If LIME fails, assign zero saliency to all tokens
                for token_id in token_ids:
                    token_id = int(token_id)
                    token_saliency = {'token': tokenizer.ids_to_tokens[token_id]}
                    for cls_ in range(args["labels"]):
                        token_saliency[int(cls_)] = 0
                    saliencies.append(token_saliency)

                # Write saliency results to the file
                out.write(json.dumps({'tokens': saliencies}) + '\n')
                out.flush()
                continue
            
            end = time.time()  # Track the end time for processing
            saliency_flops.append(end - start)  # Store time taken for saliency generation

            # Serialize the LIME explanation
            explanation = {}
            for cls_ in range(args["labels"]):
                cls_expl = {}
                for (w, s) in exp.as_list(label=cls_):  # Get feature importance scores
                    cls_expl[int(w)] = s
                explanation[cls_] = cls_expl

            # For each token, store the saliency score for each class
            for token_id in token_ids:
                token_id = int(token_id)
                token_saliency = {'token': tokenizer.ids_to_tokens[token_id]}
                for cls_ in range(args["labels"]):
                    token_saliency[int(cls_)] = explanation[cls_].get(token_id, None)
                saliencies.append(token_saliency)

            # Write the saliency results to the file
            out.write(json.dumps({'tokens': saliencies}) + '\n')
            out.flush()

    return saliency_flops  # Return the FLOPS data for performance analysis

# Configuration for the experiment
args = {
    "dataset": "snli",
    "dataset_dir": "data/e-SNLI/dataset/",
    "split": "test",
    "model": "cnn",
    "models_path": ["data/models/snli/cnn/cnn", "data/models/snli/random_cnn/cnn"],
    "gpu": False,
    "gpu_id": 0,
    "seed": 73,
    "output_dir": ["data/saliency/snli/cnn/", "data/saliency/snli/random_cnn/"],
    "labels": 3
}

# Set random seeds for reproducibility
random.seed(args["seed"])
torch.manual_seed(args["seed"])
torch.cuda.manual_seed_all(args["seed"])
torch.backends.cudnn.deterministic = True
np.random.seed(args["seed"])

# Set the device (GPU or CPU)
device = torch.device("cuda") if args["gpu"] else torch.device("cpu")

# Initialize the tokenizer (using BERT)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Loop through different models and generate saliency for each one
for model in range(1, 6):
    for models_path, output_dir in zip(args["models_path"], args["output_dir"]):
        model_path = models_path + f"_{model}"
        print(model_path, flush=True)
        # Generate and store saliency results
        all_flops = generate_saliency(model_path, os.path.join(output_dir, f'{model_path.split("/")[-1]}_lime'), args)
        # Print average and standard deviation of FLOPS
        print('FLOPS', np.average(all_flops), np.std(all_flops))
