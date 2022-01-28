import os, json, sys
import glob
from transformers import AutoModelForMaskedLM,AutoTokenizer

# 下载的模型
output_dir = "robert_large"
pretrained_model_name_or_path = "hfl/chinese-roberta-wwm-ext-large"
if len(sys.argv) == 3:
    output_dir = sys.argv[1]
    pretrained_model_name_or_path = sys.argv[2]

output_dir = "../resources/{}/".format(output_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# 下载文件
model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path, cache_dir=output_dir)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, cache_dir=output_dir)

# 重命名文件
for filename in glob.glob(output_dir + "*.json"):
    info = json.load(open(filename))
    rename = info.get("url").split("/")[-1]
    if os.path.exists(filename[:-5]):
        os.rename(filename[:-5], output_dir + rename)
    if os.path.exists(filename[:-5]+".lock"):
        os.remove(filename[:-5]+".lock")
    os.remove(filename)

