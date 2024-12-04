"""Computing Faithfulness measure for the saliency scores."""
import os
import random
from functools import partial

import numpy as np
import torch
from sklearn.metrics import auc
from transformers import BertTokenizer

from models.train_cnn import eval_model  # Import evaluation function for the CNN model
from models.data_loader import BucketBatchSampler, DatasetSaliency, collate_threshold, collate_nli, NLIDataset  # Import necessary data utilities
from models.model_builder import CNN_MODEL  # Import the CNN model builder

# Function to load a model from a given checkpoint path
def get_model(model_path):
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model_args = checkpoint['args']

    # Build the model and load the state dict
    model_cp = CNN_MODEL(tokenizer, model_args, n_labels=checkpoint['args']['labels']).to(device)
    model_cp.load_state_dict(checkpoint['model'])

    return model_cp, model_args

# Main execution block
if __name__ == "__main__":
    # Define arguments for the experiment, including paths, model configurations, and settings
    args = {
        "gpu": False,  # Whether to use GPU
        "saliency": ["rand", "shap", "sal_mean", "sal_l2", "occlusion_none", "lime", "inputx_mean", "inputx_l2", "guided_mean", "guided_l2"],  # List of saliency types to evaluate
        "dataset": "snli",  # Dataset to use (SNLI)
        "dataset_dir": "data/e-SNLI/dataset",  # Path to dataset
        "test_saliency_dir": ["data/saliency/snli/cnn/", "data/saliency/snli/random_cnn/"],  # Path to test saliency files
        "model_path": ["data/models/snli/cnn/cnn", "data/models/snli/random_cnn/cnn"],  # Path to trained models
        "models_dir": ["data/models/snli/cnn/", "data/models/snli/random_cnn/"],  # Directory containing models
        "model": "cnn",  # Model type
        "output_dir": ["data/evaluations/snli/cnn/", "data/evaluations/snli/random_cnn/"],  # Output directory for results
    }

    # Set the device to GPU if available, otherwise CPU
    device = torch.device("cuda") if args["gpu"] else torch.device("cpu")

    # Load the BERT tokenizer for text processing
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Define thresholds for evaluating saliency faithfulness
    thresholds = list(range(0, 110, 10))
    aucs = []  # List to store AUC scores for each model

    # Set the collate function for batching based on NLI dataset
    coll_call = collate_nli
    eval_fn = eval_model  # Evaluation function to use

    # Loop through each saliency method
    for saliency in args["saliency"]:
        for models_dir, test_saliency_dir, output_dir in zip(args["models_dir"], args["test_saliency_dir"], args["output_dir"]):
            # Iterate through each model in the model directory
            for model_path in os.listdir(models_dir):
                # Skip prediction files (they are not models)
                if model_path.endswith('.predictions'):
                    continue
                print('Model', model_path, flush=True)

                # Construct full model path and load the model
                model_full_path = os.path.join(models_dir, model_path)
                model, model_args = get_model(model_full_path)

                # Set random seeds for reproducibility
                random.seed(model_args["seed"])
                torch.manual_seed(model_args["seed"])
                torch.cuda.manual_seed_all(model_args["seed"])
                torch.backends.cudnn.deterministic = True
                np.random.seed(model_args["seed"])

                model_scores = []  # List to store model scores for each threshold
                # Loop through all threshold values
                for threshold in thresholds:
                    # Define the custom collate function for batching with a threshold
                    collate_fn = partial(collate_threshold,
                                         tokenizer=tokenizer,
                                         device=device,
                                         return_attention_masks=False,
                                         pad_to_max_length=False,
                                         threshold=threshold,
                                         collate_orig=coll_call,
                                         n_classes=3)

                    # Construct the path for saliency test data
                    saliency_path_test = os.path.join(test_saliency_dir, f'{model_path}_{saliency}')

                    # Load the test dataset and create a DatasetSaliency object
                    test = NLIDataset(args["dataset_dir"], type="test")
                    test = DatasetSaliency(test, saliency_path_test)

                    # Create a batch sampler for the test dataset
                    test_dl = BucketBatchSampler(batch_size=model_args["batch_size"],
                                                 dataset=test,
                                                 collate_fn=collate_fn)

                    # Evaluate the model using the eval function
                    results = eval_fn(model, test_dl, model_args["labels"])
                    model_scores.append(results[2])  # Store model score (AUC or other)

                # Print threshold and corresponding model scores
                print(thresholds, model_scores)
                aucs.append(auc(thresholds, model_scores))  # Compute AUC score for the model

            # Print the mean and standard deviation of the AUC scores
            print(f'{np.mean(aucs):.2f} ({np.std(aucs):.2f})')
            
            # Write the AUC scores to an output file
            output_file = f"{output_dir}cnn_faithfulness_{saliency}"
            with open(output_file, "w") as file:
                file.write(f"{np.mean(aucs):.2f} {np.std(aucs):.2f}\n")

