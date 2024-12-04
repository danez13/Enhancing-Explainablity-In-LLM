import os
import matplotlib
import pandas as pd

if __name__ == "__main__":
    args = {
        "eval_paths": ["data/evaluations/snli/cnn","data/evaluations/snli/random_cnn"],
        "evaluations": ["confidence","faithfulness","humanAgreement","consistency","dataConsistency"]
    }
    data = {}
    for evaluations in args["evaluations"]:
        data[evaluations] = {"mean":[],"standard deviation":[],"metrics":[]}
    for path in args["eval_paths"]:
        for eval in os.listdir(path):
            if eval == "precomp_cnn_snli_not_2_0":
                continue
            _e = eval.split("_",2)
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
    df = pd.DataFrame(data)
    print(df,flush=True)
