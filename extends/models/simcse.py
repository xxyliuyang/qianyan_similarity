from typing import Any, Dict
import torch
import torch.nn.functional as F
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import BertPooler
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides

@Model.register("simcse")
class SimilarBert(Model):

    def __init__(
        self,
        vocab: Vocabulary,
        model_path: str = "bert",
        **kwargs
    ) -> None:
        super().__init__(vocab, **kwargs)

        self._text_field_embedder = PretrainedTransformerEmbedder(model_path)
        self._text_field_embedder = BasicTextFieldEmbedder(
            {"tokens": self._text_field_embedder})

        self._pooler = BertPooler(model_path, dropout=0.1)
        self._loss = torch.nn.CrossEntropyLoss()
        self._accuracy = CategoricalAccuracy()

    def _get_embedding(self, tokens: TextFieldTensors):
        embedding = self._text_field_embedder(tokens)
        pooled_embedding = self._pooler(embedding)
        return pooled_embedding

    def _simcse_loss(self, e1, e2, e3):
        # 1. 计算句子之间的相似度
        embedding = torch.cat((e1, e2, e3), dim=0)
        sim = F.cosine_similarity(embedding.unsqueeze(1), embedding.unsqueeze(0), dim=-1)
        sim = sim - torch.eye(embedding.shape[0], device=embedding.device) * 1e12

        # 2.构造对比学习的label
        label = torch.arange(e1.shape[0], device=embedding.device)
        y_label = torch.cat((label+e1.shape[0], label))

        # 3.选择label对应的行
        select_row = torch.arange(e1.shape[0]*2, device=embedding.device)
        sim = torch.index_select(sim, dim=0, index=select_row)
        sim = sim / 0.05

        # 4.计算loss
        loss = self._loss(sim, y_label)
        return sim, y_label, loss

    def forward(self,
                text1: TextFieldTensors,
                text2: TextFieldTensors,
                text3: TextFieldTensors,
                ) -> Dict[str, torch.Tensor]:

        embedding1 = self._get_embedding(text1)
        embedding2 = self._get_embedding(text2)
        embedding3 = self._get_embedding(text3)

        sim, label, loss = self._simcse_loss(embedding1, embedding2, embedding3)
        self._accuracy(sim, label)

        result = {"logits": sim,
                  "loss": loss}

        return result

    def get_metrics(self, reset: bool = False) -> Dict[str, Dict[str, Any]]:
        return {
            "accuracy": self._accuracy.get_metric(reset),
        }


