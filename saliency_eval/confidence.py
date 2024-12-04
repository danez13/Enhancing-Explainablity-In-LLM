"""Evaluate confidence measure."""
import json
import os
import random
from collections import defaultdict

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error, mean_absolute_error
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# Function to sample the dataset in different ways (up, down, or mid)
def sample(X, y, mode='up'):
    # Create dictionaries to store indices of samples by their class bucket
    buckets_idx = defaultdict(lambda: [])
    buckets_size = defaultdict(lambda: 0)
    
    # Group indices by their class (scaled to range [0, 10])
    for i, _y in enumerate(y):
        buckets_size[int(_y * 10)] += 1
        buckets_idx[int(_y * 10)].append(i)

    # Set the sample size based on the mode
    if mode == 'up':
        sample_size = max(list(buckets_size.values()))  # Upsample to the largest class size
    elif mode == 'down':
        sample_size = min(list(buckets_size.values()))  # Downsample to the smallest class size
    elif mode == 'mid':
        sample_size = (max(list(buckets_size.values())) - min(list(buckets_size.values()))) // 2  # Midpoint between max and min

    new_idx = []

    # Sample from each class bucket
    for _, bucket_ids in buckets_idx.items():
        do_replace = True
        if sample_size <= len(bucket_ids):
            do_replace = False  # No replacement if there are enough samples in the bucket
        chosen = np.random.choice(bucket_ids, sample_size, replace=do_replace)
        new_idx += chosen.tolist()

    random.shuffle(new_idx)  # Shuffle the resulting indices

    # Return the sampled X and y
    return X[new_idx], y[new_idx]


if __name__ == "__main__":
    # Arguments for directories and saliency types
    args = {
        "models_dir": ["data/models/snli/cnn/","data/models/snli/random_cnn/"],
        "saliency_dir": ["data/saliency/snli/cnn/","data/saliency/snli/random_cnn/"],
        "saliency": ["rand","shap", "sal_mean", "sal_l2", "occlusion_none", "lime", "inputx_mean", "inputx_l2", "guided_mean", "guided_l2"],
        "upsample": "up",
        "output_dir": ["data/evaluations/snli/cnn/","data/evaluations/snli/random_cnn/"]
    }

    np.random.seed(1)  # Set random seed for reproducibility

    print(args, flush=True)  # Print the configuration
    all_y = []  # List to store all true labels
    for saliency in args["saliency"]:
        print(saliency)  # Print the current saliency method
        test_scores = []  # List to store test scores for each saliency
        test_coefs = []  # List to store coefficients for each test split

        # Loop over the model, saliency, and output directories
        for models_dir, saliency_dir, output_dir in zip(args["models_dir"], args["saliency_dir"], args["output_dir"]):
            for model_path in os.listdir(models_dir):

                # Skip files that are not prediction files
                if model_path.endswith('.predictions'):
                    continue
                print(model_path)

                full_model_path = os.path.join(models_dir, model_path)
                predictsions_path = full_model_path + '.predictions'
                saliency_path = os.path.join(saliency_dir, f"{model_path}_{saliency}")

                # Load predictions and saliency data
                predictions = json.load(open(predictsions_path))
                class_preds = predictions['class']
                logits = predictions['logits']
                saliencies = []
                all_confidences = []
                all_saliencies = []
                classes = [0, 1, 2]  # Assuming 3 classes (could be changed)
                features = []
                y = []
                tokens = []

                # Process saliency data
                with open(saliency_path) as out:
                    for i, line in enumerate(out):
                        try:
                            instance_saliency = json.loads(line)
                        except:
                            continue

                        instance_sals = []
                        instance_tokens = []
                        for _cls in classes:
                            cls_sals = []
                            for _token in instance_saliency['tokens']:
                                if _cls == 0:
                                    instance_tokens.append(_token['token'])
                                if _token['token'] == '[PAD]':
                                    break
                                cls_sals.append(_token[str(_cls)])
                            instance_sals.append(cls_sals)
                        saliencies.append(instance_sals)
                        tokens.append(instance_tokens)

                # Extract features based on saliency and confidence
                for i, instance in enumerate(saliencies):
                    _cls = class_preds[i]
                    instance_saliency = saliencies[i]
                    instance_logits = softmax(logits[i])  # Apply softmax to logits

                    confidence_pred = instance_logits[_cls]  # The model's confidence in its prediction
                    saliency_pred = np.array(instance_saliency[_cls])

                    left_classes = classes.copy()
                    left_classes.remove(_cls)
                    other_sals = [np.array(instance_saliency[c_]) for c_ in left_classes]
                    feats = []

                    # Compute features based on saliency differences
                    if len(classes) == 2:
                        feats.append(sum(saliency_pred - other_sals[0]))
                        feats.append(sum(saliency_pred - other_sals[0]))
                        feats.append(sum(saliency_pred - other_sals[0]))
                    else:
                        feats.append(sum(np.max([saliency_pred - other_sals[0], saliency_pred - other_sals[1]], axis=0)))
                        feats.append(sum(np.mean([saliency_pred - other_sals[0], saliency_pred - other_sals[1]], axis=0)))
                        feats.append(sum(np.min([saliency_pred - other_sals[0], saliency_pred - other_sals[1]], axis=0)))

                    y.append(confidence_pred)  # Store the confidence
                    features.append(feats)

                # Normalize the features
                features = MinMaxScaler().fit_transform(np.array(features))
                all_y += y  # Store true labels
                y = np.array(y)

                # Perform cross-validation with ShuffleSplit
                rs = ShuffleSplit(n_splits=5, random_state=2)
                scores = []
                coefs = []
                for train_index, test_index in rs.split(features):
                    X_train, y_train, X_test, y_test = features[train_index], y[train_index], features[test_index], y[test_index]

                    # Upsample the training data if specified
                    if args["upsample"] == 'up':
                        X_train, y_train = sample(X_train, y_train, mode='up')

                    reg = LinearRegression().fit(X_train, y_train)  # Fit a linear regression model
                    pred = reg.predict(X_train)  # Predictions on training data
                    test_pred = reg.predict(X_test)  # Predictions on test data

                    # Evaluate the model with mean absolute error and max error
                    all_metrics = []
                    for metric in [mean_absolute_error, max_error]:
                        all_metrics.append(metric(y_test, test_pred))
                    scores.append(all_metrics)
                    coefs.append(reg.coef_)

                # Store the test scores
                test_scores.append([np.mean([_s[i] for _s in scores]) for i in range(len(scores[0]))])

            # Print the mean and standard deviation of test scores
            print(' '.join([f"{np.mean([_s[l] for _s in test_scores]):.3f} "
                            f"($\pm$ {np.std([_s[l] for _s in test_scores]):.3f})"
                            for l in range(len(test_scores[0]))]), flush=True)
            
            # Write the results to the output file
            output_file = f"{output_dir}cnn_confidence_{saliency}"
            with open(output_file, "w") as file:
                for l in range(len(test_scores[0])):
                    file.write(f"{np.mean([_s[l] for _s in test_scores]):.3f} {np.std([_s[l] for _s in test_scores]):.3f}\n")
