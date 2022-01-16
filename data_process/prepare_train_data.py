import os
from tools.file_util import load_dataset, save_json

if __name__ == '__main__':
    dirs = ['bq_corpus', 'lcqmc', 'paws-x-zh']
    subsets = ["train", "dev", "test"]

    for dir in dirs:
        datasets = load_dataset(dir, subsets=subsets)
        print("dataset={}, train-dev-test size={}-{}-{}.".format(dir, len(datasets[0]), len(datasets[1]), len(datasets[2])))

        filedir = "data/trainset/" + dir + "/"
        if not os.path.exists(filedir):
            os.mkdir(filedir)

        for subset, dataset in zip(subsets, datasets):
            filename = filedir + subset + ".json"
            save_json(dataset, filename)
        print("sava {} done.\n".format(dir))
