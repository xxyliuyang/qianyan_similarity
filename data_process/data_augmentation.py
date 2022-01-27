import os
from tools.file_util import load_json, save_json

def swap_aug(data):
    result = []
    for case in data:
        case['text1'], case['text2'] = case['text2'], case['text1']
        result.append(case)
    return result

def closure_aug(data):
    result = []
    combine = {}
    for case in data:
        if case['label'] != "1":
            continue
        if case['text1'] not in combine:
            combine[case['text1']] = []
        if case['text2'] not in combine:
            combine[case['text2']] = []

        combine[case['text1']].append(case['text2'])
        combine[case['text2']].append(case['text1'])

    for key, value in combine.items():
        if len(value) <= 1:
            continue
        for i in range(len(value)):
            for j in range(len(value)):
                if i==j:
                    continue
                case = {
                    "text1": value[i],
                    "text2": value[j],
                    "label": "1",
                }
                result.append(case)
    return result

def merge_result(all_data):
    pairset = set([])
    result = []
    for data in all_data:
        for case in data:
            pair = case['text1'] + case['text2']
            if pair in pairset and case['label'] == "1":
                continue
            result.append(case)
    return result


def augmentation(data):
    aug1 = swap_aug(data)
    aug2 = closure_aug(data)
    print("aug1={}, aug2={}".format(len(aug1), len(aug2)))
    result = merge_result([data, aug1, aug2])
    return result

if __name__ == '__main__':
    dirs = ['bq_corpus', 'lcqmc', 'paws-x-zh']
    subsets = ["train"]

    for dir in dirs:
        filedir = "data/trainset/" + dir + "/"

        for subset in subsets:
            filename = filedir + subset + ".json"
            data = load_json(filename)
            print("load data: ", len(data))
            data = augmentation(data)
            print("augmentation data: ", len(data))
            save_json(data, filename)
        print("sava {} done.\n".format(dir))
