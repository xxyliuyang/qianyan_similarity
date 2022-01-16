

def load_file(filename):
    data = []
    with open(filename) as fin:
        for line in fin:
            line = line.strip().split("\t")
            if "test" not in filename and len(line) != 3:
                continue
            case = {
                "text1": line[0],
                "text2": line[1],
            }
            if len(line) == 3:
                case['label'] = line[2]
            data.append(case)
    return data


def load_dataset(dir):
    file_dir = "data/original/" + dir + "/"
    subsets = ["train", "dev", "test"]

    dataset = []
    for subset in subsets:
        filename = file_dir + subset + ".tsv"
        dataset.append(load_file(filename))
    return dataset