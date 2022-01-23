from typing import List
from allennlp.predictors import Predictor
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from overrides import overrides
from tqdm import trange
import math

from tools.file_util import load_json

from extends.model import SimilarBert
from extends.reader import SimilarReader

@Predictor.register("similar")
class SimilarPredictor(Predictor):

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(json_dict)

    def predict(self, data: List) -> List[JsonDict]:
        return self.predict_batch_json(data)


class Calculator(object):
    def __init__(self,
                 data_name,
                 exp_names,
                 batch_size: int = 32,
                 device: int = -1):
        self.batch_size = batch_size
        self.device = device
        self._init_predictors(data_name, exp_names)

    def _init_predictors(self, data_name, exp_names):
        self.predictors = []
        for exp_name in exp_names:
            model_path = "records/{}/{}/model.tar.gz".format(data_name, exp_name)
            self.predictors.append(Predictor.from_path(model_path, "similar", cuda_device=self.device))

    def get_unsemble(self, batch):
        if len(self.predictors) == 1:
            return self.predictors[0].predict(batch)

    def get_predictions(self, test_data):
        # 获取预测结果
        result = []
        batch_number = math.ceil(len(test_data) / self.batch_size)
        for idx in trange(batch_number, ncols=70, desc="[Predict]:"):
            index1 = idx * self.batch_size
            index2 = min((idx + 1) * self.batch_size, len(test_data))
            batch = test_data[index1: index2]

            res = self.get_unsemble(batch)
            result += res
        return result

    def eval(self, data, result, output_file):
        fout = open(output_file, 'w')
        for d, p in zip(data, result):
            if d['label'] != str(p['label']):
                out_info = []
                out_info.append(d['text1'])
                out_info.append(d['text2'])
                out_info.append(d['label'])
                out_info.append("pred={}. probs={}\n\n".format(p['label'], p['label_probs']))
                fout.write("\n".join(out_info))

    def test(self, data, result, output_file):
        fout = open(output_file, 'w')
        fout.write("{}\t{}\n".format("index", "prediction"))
        for i,(d, p) in enumerate(zip(data, result)):
            fout.write("{}\t{}\n".format(i, p['label']))

    def get_output(self, filename, output_file, is_test=False):
        # 1. load数据
        data = load_json(filename)
        # 2. predict
        result = self.get_predictions(data)
        # 3. 输出
        if is_test:
            self.test(data, result, output_file)
        else:
            self.eval(data, result, output_file)


if __name__ == '__main__':
    exp_names = ["base"]
    # data_names = ["bq_corpus", "lcqmc", "paws-x-zh"]
    data_names = ["demo"]
    subset = "dev"
    device = -1

    for data_name in data_names:
        calculator = Calculator(data_name, exp_names, device=device)
        filename = "data/trainset/{}/{}.json".format(data_name, subset)
        output_file = "records/result/{}_{}.txt".format(data_name, subset)
        calculator.get_output(filename, output_file, subset=="test")







