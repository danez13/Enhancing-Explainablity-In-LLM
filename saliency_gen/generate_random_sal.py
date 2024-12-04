"""Generates random saliency scores for a baseline."""
import argparse
import json
import random
import time
import numpy as np
import torch

if __name__ == "__main__":
    # Initialize the argument dictionary with default values
    args = {
        "saliencies": ["shap", "sal_mean", "sal_l2", "occlusion_none", "lime", "inputx_mean", "inputx_l2", "guided_mean", "guided_l2"],
        "output_path": ["data/saliency/snli/cnn/cnn", "data/saliency/snli/random_cnn/cnn"],
        "seed": 73,
        "labels": 3
    }
    classes = list(range(args["labels"]))  # List of class indices (e.g., [0, 1, 2] for 3 classes)

    flops = []  # List to track FLOP times across different saliency methods and models

    # Loop through each saliency method and model
    for sal in args["saliencies"]:
        for m in range(1, 6):  # Iterating over models (from model 1 to model 5)
            for output in args["output_path"]:
                saliency_path = f"{output}_{m}_{sal}"  # Path to the current saliency file
                output_path = f"{output}_{m}_rand"  # Output path for the random saliency scores
                print(m)

                # Set the random seed for reproducibility
                random.seed(args["seed"])
                torch.manual_seed(args["seed"])
                torch.cuda.manual_seed_all(args["seed"])
                torch.backends.cudnn.deterministic = True
                np.random.seed(args["seed"])

                # Open the saliency file and write the random saliency scores
                with open(saliency_path) as out:
                    with open(output_path, 'w') as output_sal:
                        saliency_flops = []  # List to track FLOP times for each batch

                        # Iterate through each line in the saliency file
                        for j, line in enumerate(out):
                            start = time.time()  # Start the timer

                            try:
                                instance_saliency = json.loads(line)  # Parse the JSON saliency data
                            except:
                                line = next(out)  # If there's an error, try the next line
                                instance_saliency = json.loads(line)

                            # For each token in the instance's saliency map
                            for i, token in enumerate(instance_saliency['tokens']):
                                if token['token'] == '[PAD]':  # Skip padding tokens
                                    continue
                                for _c in classes:  # Loop through all classes (0 to 2 in this case)
                                    instance_saliency['tokens'][i][str(_c)] = np.random.rand()  # Assign a random saliency score

                            output_sal.write(json.dumps(instance_saliency) + '\n')  # Write the modified saliency data

                            end = time.time()  # End the timer
                            saliency_flops.append(end - start)  # Calculate the time for this batch

                print(np.mean(saliency_flops), np.std(saliency_flops))  # Print the mean and std of FLOPs for this model and saliency method
                flops.append(np.mean(saliency_flops))  # Track the mean FLOPs across different saliency methods

            # Print the overall FLOPs for all saliency methods and models
            print('FLOPs', f'{np.mean(flops):.2f} {np.std(flops):.2f}')

