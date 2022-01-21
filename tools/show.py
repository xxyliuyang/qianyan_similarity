import os
import json
from pathlib import Path

def metric_parser(exp_dir):
    """
    解析 metrics_epoch_*.json 中的信息
    """
    info = []
    for filename in os.listdir(exp_dir):
        if not filename.startswith("metrics_epoch"):
            continue
        metric_file = exp_dir / filename
        metrics = json.load(open(metric_file))
        epoch = metrics['epoch']
        if "validation_accuracy" in metrics:
            ratio = "%.4f" % (float(metrics['validation_accuracy']))
            info.append([epoch, str(ratio)])
    info = sorted(info)
    return [k[1] for k in info]

def show_datset_result(name):
    paths = sorted(Path("records/{}/".format(name)).iterdir(), key=os.path.getmtime)
    for path in paths:
        if os.path.exists(path):
            info = metric_parser(path)
            if len(info) > 0:
                print("{}\n{}".format(path, " ".join(info)))
            print()

if __name__ == "__main__":
    for name in ["bq_corpus", "lcqmc", "paws-x-zh"]:
        show_datset_result(name)