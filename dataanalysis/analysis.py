from collections import defaultdict
from tools.file_util import load_dataset

def label_count(data):
    result = defaultdict(int)
    for case in data:
        result[case['label']] += 1
    return result

if __name__ == '__main__':
    dirs = ['bq_corpus', 'lcqmc', 'paws-x-zh']

    for dir in dirs:
        dataset = load_dataset(dir)
        print("dataset={}, train-dev-test size={}-{}-{}.".format(dir, len(dataset[0]), len(dataset[1]), len(dataset[2])))

        for i, data in enumerate(dataset[:2]):
            print("index-{}, label count={}".format(i, str(label_count(data))))
        print()