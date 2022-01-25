import os
import random
from tools.file_util import load_dataset, save_json

def _get_combine(data):
    resuts = {}
    for case in data:
        text1 = case['text1']
        text2 = case['text2']
        label = case['label']

        if text1 not in resuts:
            resuts[text1] = {}
        resuts[text1][label].append(text2)
    return resuts

def _get_simcse_example(data_combine):
    results = []
    example_set = set([])

    for text1, infos in data_combine.items():
        if "0" not in infos or "1" not in infos:
            continue

        text2 = infos["1"]
        text3 = infos["0"]
        temp1 = text1 + text2 + text3
        temp2 = text2 + text1 + text3

        if temp1 in example_set or temp2 in example_set:
            continue
        example_set.add(temp1)
        case = {
            "text1": text1,
            "text2": text2,
            "text3": text3
        }
        results.append(case)
    return results


def prepare_simcse(data):
    data_combine = _get_combine(data)
    print("data_combin={}".format(len(data_combine)))
    simcse_examples = _get_simcse_example(data_combine)
    print("simcse_examples={}".format(len(simcse_examples)))
    return simcse_examples


if __name__ == '__main__':
    dirs = ['bq_corpus', 'lcqmc', 'paws-x-zh']
    subsets = ["train", "dev"]

    filedir = "data/simcse/"
    if not os.path.exists(filedir):
        os.mkdir(filedir)

    train_data = []
    dev_data = []
    for dir in dirs:
        datasets = load_dataset(dir, subsets=subsets)
        train_data += datasets[0]
        dev_data += datasets[1]
    print("train_data={}, dev_data={}".format(len(train_data), len(dev_data)))

    train_file = filedir + "train.json"
    train_simcse_examples = prepare_simcse(train_data)
    random.shuffle(train_simcse_examples)
    save_json(train_simcse_examples, train_file)

    dev_file = filedir + "dev.json"
    dev_simcse_examples = prepare_simcse(dev_data)
    random.shuffle(dev_simcse_examples)
    save_json(dev_simcse_examples, dev_file)


