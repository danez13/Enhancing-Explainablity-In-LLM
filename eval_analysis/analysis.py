import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == "__main__":
    args = {
        "eval_paths": ["data/evaluations/snli/cnn"],
        "evaluations": ["confidence","faithfulness","humanAgreement","consistency","dataConsistency"],
        "save_figures": "data/analysis/"
    }
    data = {}
    for evaluations in args["evaluations"]:
        data[evaluations] = {"mean":[],"standard deviation":[],"metrics":[]}
    for path in args["eval_paths"]:
        for eval in os.listdir(path):
            if eval == "precomp_cnn_snli_not_2_0":
                continue
            _e = eval.split("_",2)
            if "random" in path:
                _e[2] = f"random_{_e[2]}"
            if _e[2] == "rand":
                continue
            for line in open(f"{path}/{eval}"):
                newline = line.replace("\n","")
                values = newline.split(" ")
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
    for key, _d in data.items():
        print(key)
        output_file = f"{args['save_figures']}{key}.png"
        df = pd.DataFrame(_d)
        df["mean"] = pd.to_numeric(df["mean"], errors='coerce')
        df["standard deviation"] = pd.to_numeric(df["standard deviation"], errors='coerce')
        # Now plot the data
        df.plot(x="metrics", y=["mean", "standard deviation"], kind="bar", figsize=(12, 15))
        plt.title(f"Mean and Standard Deviation for {key}")
        plt.xlabel("Metrics")
        plt.ylabel("Values")
        plt.savefig(output_file)
