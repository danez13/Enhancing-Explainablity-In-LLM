"""Generates random saliency scores for a baseline."""
import argparse
import json
import random
import time
import numpy as np
import torch

if __name__ == "__main__":
    args = {
        "saliencies": ["shap", "sal_mean", "sal_l2", "occlusion_none", "lime", "inputx_mean", "inputx_l2", "guided_mean", "guided_l2"],
        "output_path": ["data/saliency/snli/cnn/cnn","data/saliency/snli/random_cnn/cnn"],
        "seed": 73,
        "labels": 3
    }
    classes = list(range(args["labels"]))

    flops = []
    for sal in args["saliencies"]:
        for m in range(1,6):
            for output in args["output_path"]:
                saliency_path = f"{output}_{m}_{sal}"
                output_path = f"{output}_{m}_rand"
                print(m)

                random.seed(args["seed"])
                torch.manual_seed(args["seed"])
                torch.cuda.manual_seed_all(args["seed"])
                torch.backends.cudnn.deterministic = True
                np.random.seed(args["seed"])

                with open(saliency_path) as out:
                    with open(output_path, 'w') as output_sal:
                        saliency_flops = []
                        for j, line in enumerate(out):
                            start = time.time()

                            try:
                                instance_saliency = json.loads(line)
                            except:
                                line = next(out)
                                instance_saliency = json.loads(line)

                            for i, token in enumerate(instance_saliency['tokens']):
                                if token['token'] == '[PAD]':
                                    continue
                                for _c in classes:
                                    instance_saliency['tokens'][i][
                                        str(_c)] = np.random.rand()

                            output_sal.write(json.dumps(instance_saliency) + '\n')

                            end = time.time()
                            saliency_flops.append(end-start)

                print(np.mean(saliency_flops), np.std(saliency_flops))
                flops.append(np.mean(saliency_flops))

            print('FLOPs', f'{np.mean(flops):.2f} ($\pm${np.std(flops):.2f})')