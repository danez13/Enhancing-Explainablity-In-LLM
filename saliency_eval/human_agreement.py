"""Computing Human Agreement measure."""
import argparse
import json

import numpy as np
from sklearn.metrics import average_precision_score
from transformers import BertTokenizer
from models.data_loader import NLIDataset
from models.saliency_utils import get_gold_saliency_esnli

if __name__ == "__main__":
    args= {
        "subset": "all",
        "dataset": "snli",
        "dataset_dir": "data/e-SNLI/dataset",
        "saliency_path": ["data/saliency/snli/cnn/cnn","data/saliency/snli/random_cnn/cnn"],
        "saliencies": ["rand","shap", "sal_mean", "sal_l2", "occlusion_none", "lime", "inputx_mean", "inputx_l2", "guided_mean", "guided_l2"]
    }
    print(args, flush=True)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    test = test =NLIDataset(args["dataset_dir"], type='test',salient_features=True)

    for saliency_name in args["saliencies"]:
        avg_seeds = []
        for m in range(1,6):
            for sp in args["saliency_path"]:
                saliency_path = sp+f"{m}"
                avgp = []

                prediction_path = saliency_path.replace('saliency',
                                                        'models') + '.predictions'
                predictions = json.load(open(prediction_path))['class']
                saliency_path = saliency_path + '_' + saliency_name
                with open(saliency_path) as out:
                    for i, line in enumerate(out):
                        try:
                            instance_saliency = json.loads(line)
                        except:
                            line = next(out)
                            instance_saliency = json.loads(line)
                        saliency = instance_saliency['tokens']

                        instance = test[i]
                        instance_gold = instance[2]
                        predicted = predictions[i]

                        token_ids = tokenizer.encode(instance[0], instance[1])

                        token_pred_saliency = []
                        for record in saliency:
                            token_pred_saliency.append(record[str(instance_gold)])

                        gold_saliency = get_gold_saliency_esnli(instance,
                                                        tokenizer.convert_ids_to_tokens(
                                                            token_ids),
                                                        [tokenizer.cls_token,
                                                        tokenizer.sep_token,
                                                        tokenizer.pad_token],
                                                        tokenizer)

                        gold_saliency = gold_saliency[:509]
                        token_pred_saliency = token_pred_saliency[
                                            :len(gold_saliency)]

                        avgp.append(average_precision_score(gold_saliency,
                                                            token_pred_saliency))

                    print(len(avgp), np.mean(avgp), flush=True)
                    avg_seeds.append(np.mean(avgp))
            print(saliency_name, flush=True)
            print(f'{np.mean(avg_seeds):.3f} ($\pm${np.std(avg_seeds):.3f})',
                flush=True)
