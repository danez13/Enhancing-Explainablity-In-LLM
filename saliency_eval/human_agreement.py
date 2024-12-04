"""Computing Human Agreement measure."""
import json

import numpy as np
from sklearn.metrics import average_precision_score
from transformers import BertTokenizer
from models.data_loader import NLIDataset
from models.saliency_utils import get_gold_saliency_esnli

# Main execution block
if __name__ == "__main__":
    # Define arguments for paths and configurations
    args = {
        "subset": "all",  # Use all data
        "dataset": "snli",  # Use SNLI dataset
        "dataset_dir": "data/e-SNLI/dataset",  # Path to dataset
        "saliency_path": ["data/saliency/snli/cnn/cnn", "data/saliency/snli/random_cnn/cnn"],  # Saliency paths
        "saliencies": ["rand", "shap", "sal_mean", "sal_l2", "occlusion_none", "lime", "inputx_mean", "inputx_l2", "guided_mean", "guided_l2"],  # List of saliency types to evaluate
        "output_dir": ["data/evaluations/snli/cnn/", "data/evaluations/snli/random_cnn/"],  # Output directories for results
    }
    print(args, flush=True)

    # Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load the e-SNLI test dataset
    test = NLIDataset(args["dataset_dir"], type='test', salient_features=True)

    # Loop over each saliency method
    for saliency_name in args["saliencies"]:
        avg_seeds = []  # List to store average human agreement scores for each seed
        for m in range(1, 6):  # Loop over 5 different seeds (m = 1 to 5)
            # Loop over each saliency path and corresponding output directory
            for sp, output_dir in zip(args["saliency_path"], args["output_dir"]):
                saliency_path = sp + f"_{m}"  # Construct the path for this seed
                avgp = []  # List to store average precision scores for each instance

                # Construct the path to the model's predictions
                prediction_path = saliency_path.replace('saliency', 'models') + '.predictions'
                predictions = json.load(open(prediction_path))['class']  # Load model predictions
                saliency_path = saliency_path + '_' + saliency_name  # Add saliency type to the path
                
                # Open the saliency file and process it
                with open(saliency_path) as out:
                    for i, line in enumerate(out):
                        try:
                            # Try to load saliency for the instance
                            instance_saliency = json.loads(line)
                        except:
                            # If failed, try loading the next line
                            line = next(out)
                            instance_saliency = json.loads(line)
                        saliency = instance_saliency['tokens']  # Extract token saliency information

                        # Get the current instance from the test set
                        instance = test[i]
                        instance_gold = instance[2]  # Get the gold label for this instance
                        predicted = predictions[i]  # Get the predicted class for this instance

                        # Tokenize the instance using BERT tokenizer
                        token_ids = tokenizer.encode(instance[0], instance[1])

                        # Prepare the saliency data for the predicted class
                        token_pred_saliency = []
                        for record in saliency:
                            token_pred_saliency.append(record[str(instance_gold)])

                        # Get the gold saliency using the utility function
                        gold_saliency = get_gold_saliency_esnli(
                            instance,
                            tokenizer.convert_ids_to_tokens(token_ids),
                            [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token],
                            tokenizer
                        )

                        # Trim gold saliency to the max sequence length
                        gold_saliency = gold_saliency[:509]
                        token_pred_saliency = token_pred_saliency[:len(gold_saliency)]

                        # Calculate the average precision score for this instance
                        avgp.append(average_precision_score(gold_saliency, token_pred_saliency))

                    # Print the number of instances and the average score
                    print(len(avgp), np.mean(avgp), flush=True)
                    avg_seeds.append(np.mean(avgp))  # Store the average score for this seed

                # Print the results for the current saliency method
                print(saliency_name, flush=True)
                print(f'{np.mean(avg_seeds):.3f} ($\pm${np.std(avg_seeds):.3f})', flush=True)
                
                # Write the average precision and standard deviation to an output file
                output_file = f"{output_dir}cnn_humanAgreement_{saliency_name}"
                with open(output_file, "w") as file:
                    file.write(f"{np.mean(avg_seeds):.3f} {np.std(avg_seeds):.3f}\n")

