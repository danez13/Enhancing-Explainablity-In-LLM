"""Script for training LSTM and CNN models for the e-SNLI dataset."""
import random
from functools import partial
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, \
    precision_recall_fscore_support
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import BatchSampler
from tqdm import tqdm
from transformers import BertTokenizer

# Importing required custom modules for data loading and model building
from models.data_loader import BucketBatchSampler, NLIDataset, collate_nli
from models.model_builder import CNN_MODEL, EarlyStopping

# Suppress warnings from the torch.utils.data.sampler module
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.sampler")

# Function to train the model for a specified number of epochs
def train_model(model: torch.nn.Module,
                train_dl: BatchSampler, dev_dl: BatchSampler,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.LambdaLR,
                n_epochs: int,
                early_stopping: EarlyStopping) -> (Dict, Dict):

    loss_f = torch.nn.CrossEntropyLoss()

    best_val, best_model_weights = {'val_f1': 0}, None  # Initialize best performance tracking

    # Loop through the specified number of epochs
    for ep in range(n_epochs):
        model.train()  # Set the model to training mode
        for batch in tqdm(train_dl, desc='Training'):  # Loop through batches in the training data
            optimizer.zero_grad()  # Clear previous gradients
            logits = model(batch[0])  # Forward pass through the model
            loss = loss_f(logits, batch[1])  # Calculate the loss
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update model parameters

        # Evaluate the model on the validation dataset after each epoch
        val_p, val_r, val_f1, val_loss, _, _ = eval_model(model, dev_dl)
        current_val = {
            'val_p': val_p, 'val_r': val_r, 'val_f1': val_f1,
            'val_loss': val_loss, 'ep': ep
        }

        # Print the validation performance for the current epoch
        print(current_val, flush=True)

        # Save the model weights if the current validation F1 score is the best
        if current_val['val_f1'] > best_val['val_f1']:
            best_val = current_val
            best_model_weights = model.state_dict()

        # Adjust learning rate based on validation loss
        scheduler.step(val_loss)
        if early_stopping.step(val_f1):
            print('Early stopping...')
            break


    return best_model_weights, best_val

# Function to evaluate the model on a given dataset (e.g., validation or test)
def eval_model(model: torch.nn.Module, test_dl: BucketBatchSampler,
               measure=None):
    model.eval()  # Set the model to evaluation mode

    loss_f = torch.nn.CrossEntropyLoss()  # Cross-entropy loss function

    with torch.no_grad():  # Disable gradient calculation during evaluation
        labels_all = []
        logits_all = []
        losses = []
        for batch in tqdm(test_dl, desc="Evaluation"):  # Loop through batches in the dataset
            logits_val = model(batch[0])  # Forward pass through the model
            loss_val = loss_f(logits_val, batch[1])  # Calculate the loss
            losses.append(loss_val.item())  # Record the loss for averaging

            labels_all += batch[1].detach().cpu().numpy().tolist()  # Collect true labels
            logits_all += logits_val.detach().cpu().numpy().tolist()  # Collect model predictions

        prediction = np.argmax(np.array(logits_all), axis=-1)  # Get predicted class labels

        # Measure evaluation metric (accuracy or F1 score)
        if measure == 'acc':
            p, r = None, None
            f1 = accuracy_score(labels_all, prediction)  # Compute accuracy
        else:
            p, r, f1, _ = precision_recall_fscore_support(labels_all,
                                                          prediction,
                                                          average='macro')  # Compute F1 score

        print(confusion_matrix(labels_all, prediction))  # Print confusion matrix

    return p, r, f1, np.mean(losses), labels_all, prediction

# Main function for model setup and training/testing
if __name__ == "__main__":
    # Loop through different model paths for multiple experiments
    for i in range(1,6):
        args = {
            "gpu":False,  # Use CPU (set to True for GPU)
            "init_only": False,  # Whether to initialize model from scratch
            "seed":73,  # Random seed for reproducibility
            "labels":3,  # Number of classes for classification
            "dataset_dir":"data/e-SNLI/dataset",  # Path to dataset
            "model_path": [f"data/models/snli/cnn/cnn_{i}",f"data/models/snli/random_cnn/cnn_{i}"],  # Paths to save models
            "batch_size": 256,  # Batch size for training
            "lr":0.0001,  # Learning rate for optimizer
            "epochs":100,  # Number of training epochs
            "mode": "test",  # Mode (e.g., train or test)
            "patience": 5,  # Early stopping patience
            "model": "cnn",  # Model type (CNN in this case)
            "embedding_dir": "./glove/",  # Directory for word embeddings
            "dropout":0.05,  # Dropout rate
            "embedding_dim":300,  # Word embedding dimension
            "in_channels":1,  # Input channels for CNN
            "out_channels": 300,  # Output channels for CNN
            "kernel_heights": [4,5,6,7],  # Kernel heights for convolution layers
            "stride":1,  # Stride for convolution layers
            "padding":0  # Padding for convolution layers
        }
        # Loop through different model paths to initialize and train models
        for index, path in enumerate(args["model_path"]):
        for index,path in enumerate(args["model_path"]):
            print(index)
            if index == 1:
                args["init_only"] = True  # If testing, only initialize the model, do not train

            # Set random seed for reproducibility
                args["init_only"]=True
            random.seed(args["seed"])
            np.random.seed(args["seed"])
            torch.manual_seed(args["seed"])
            torch.cuda.manual_seed_all(args["seed"])
            torch.backends.cudnn.deterministic = True

            # Set device (CUDA or CPU)
            device = torch.device("cuda") if args["gpu"] else torch.device("cpu")

            # Initialize the tokenizer for BERT-based models
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

            # Create a partial function for collating NLI data with tokenization
            collate_fn = partial(collate_nli, tokenizer=tokenizer, device=device,
                                 return_attention_masks=False, pad_to_max_length=False)

            # Sort key function for batching based on sentence lengths
            sort_key = lambda x: len(x[0]) + len(x[1])

            # Initialize the CNN model
            model = CNN_MODEL(tokenizer, args, n_labels=3).to(device)

            print("Loading datasets...")
            # Load the training and validation datasets
            train = NLIDataset(args["dataset_dir"], type='train', salient_features=True)
            dev = NLIDataset(args["dataset_dir"], type='dev')

            # Create data loaders for training and validation
            train_dl = BucketBatchSampler(batch_size=args["batch_size"],
                                          sort_key=sort_key, dataset=train,
                                          collate_fn=collate_fn)
            dev_dl = BucketBatchSampler(batch_size=args["batch_size"],
                                        sort_key=sort_key, dataset=dev,
                                        collate_fn=collate_fn)

            # Print model architecture
            print(model)

            # Initialize the optimizer and learning rate scheduler
            optimizer = AdamW(model.parameters(), lr=args["lr"])
            scheduler = ReduceLROnPlateau(optimizer)
            es = EarlyStopping(patience=args["patience"], percentage=False, mode='max', min_delta=0.0)

            # Train the model if not in "init_only" mode
            if not args["init_only"]:
                best_model_w, best_perf = train_model(model, train_dl, dev_dl,
                                                    optimizer, scheduler,
                                                    args["epochs"],
                                                    es)
            else:
                best_model_w, best_perf = model.state_dict(), {'val_f1': 0}

            # Save the trained model checkpoint
            checkpoint = {
                'performance': best_perf,
                'args': args,
                'model': best_model_w,
            }

            print(best_perf)  # Print best performance metrics
            print(args)  # Print the hyperparameters

            # Save the checkpoint to the specified path
            torch.save(checkpoint, path)
