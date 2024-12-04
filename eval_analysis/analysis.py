# Import necessary libraries
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Main entry point of the script
if __name__ == "__main__":
    # Define the arguments with paths to evaluation files and output settings
    args = {
        "eval_paths": ["data/evaluations/snli/cnn"],  # Paths where evaluation files are stored
        "evaluations": ["confidence", "faithfulness", "humanAgreement", "consistency", "dataConsistency"],  # Types of evaluations to process
        "save_figures": "data/analysis/"  # Directory to save the generated figures
    }
    
    # Initialize a dictionary to store data for each evaluation
    data = {}
    for evaluation in args["evaluations"]:
        data[evaluation] = {"mean": [], "standard deviation": [], "metrics": []}
    
    # Iterate through the paths to the evaluation directories
    for path in args["eval_paths"]:
        for eval in os.listdir(path):  # Loop through the evaluation files in the directory
            if eval == "precomp_cnn_snli_not_2_0":  # Skip this specific file
                continue
            # Split the file name into components
            _e = eval.split("_", 2)
            
            # Modify the evaluation label if the path contains "random"
            if "random" in path:
                _e[2] = f"random_{_e[2]}"
            
            # Skip the 'rand' model evaluation type
            if _e[2] == "rand":
                continue
            
            # Read the contents of the evaluation file
            for line in open(f"{path}/{eval}"):
                newline = line.replace("\n", "")  # Remove newline characters
                values = newline.split(" ")  # Split the values by space
                
                # Process each type of evaluation and store the mean and standard deviation
                if _e[1] == "confidence":
                    data[_e[1]]["metrics"].append(_e[2])
                    data[_e[1]]["mean"].append(values[0])
                    data[_e[1]]["standard deviation"].append(values[1])
                elif _e[1] == "faithfulness":
                    data[_e[1]]["metrics"].append(_e[2])
                    data[_e[1]]["mean"].append(values[0])
                    data[_e[1]]["standard deviation"].append(values[1])
                elif _e[1] == "humanAgreement":
                    data[_e[1]]["metrics"].append(_e[2])
                    data[_e[1]]["mean"].append(values[0])
                    data[_e[1]]["standard deviation"].append(values[1])
                elif _e[1] == "consistency":
                    data[_e[1]]["metrics"].append(_e[2])
                    data[_e[1]]["mean"].append(values[0])
                    data[_e[1]]["standard deviation"].append(values[1])
                elif _e[1] == "dataConsistency":
                    data[_e[1]]["metrics"].append(_e[2])
                    data[_e[1]]["mean"].append(values[0])
                    data[_e[1]]["standard deviation"].append(values[1])
    
    # Iterate over each evaluation type and generate the corresponding bar plot
    for key, _d in data.items():
        print(key)  # Print the evaluation type
        
        # Define the output file name based on the evaluation type
        output_file = f"{args['save_figures']}{key}.png"
        
        # Create a DataFrame from the data dictionary for the current evaluation type
        df = pd.DataFrame(_d)
        
        # Convert 'mean' and 'standard deviation' columns to numeric, ignoring errors
        df["mean"] = pd.to_numeric(df["mean"], errors='coerce')
        df["standard deviation"] = pd.to_numeric(df["standard deviation"], errors='coerce')
        
        # Plot the data as a bar plot
        df.plot(x="metrics", y=["mean", "standard deviation"], kind="bar", figsize=(12, 15))
        
        # Add title and labels to the plot
        plt.title(f"Mean and Standard Deviation for {key}")
        plt.xlabel("Metrics")
        plt.ylabel("Values")
        
        # Save the generated plot as a PNG file
        plt.savefig(output_file)

