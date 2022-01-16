import json
import shutil
from allennlp.commands import main
import sys

device = -1
dataname = "demo"

exp_name = "base"
force = "force"


# 指定训练的 config，output_dir
train_data ='./data/trainset/{}/train.json'.format(dataname);
dev_data = './data/trainset/{}/dev.json'.format(dataname);
config_file = "experiments/base.jsonnet"
overrides = json.dumps({"train_data_path": train_data,
                        "validation_data_path": dev_data,
                        "trainer": {"cuda_device": device}})
serialization_dir = "records/" + exp_name

# 是否覆盖 output_dir 文件夹：force 参数
assert force in ["force", "not_force"], "Please confirm whether to overwrite the output folder."
if force == "force":
    shutil.rmtree(serialization_dir, ignore_errors=True)


# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "--include-package", "extends", # 模型的扩展包（路径）
    "-o", overrides, # 覆盖掉 config 中的参数
     "--file-friendly-logging",
    "-s", serialization_dir
]

main()