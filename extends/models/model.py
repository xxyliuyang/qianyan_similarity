from typing import Any, Dict, Optional
import torch
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import BertPooler
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides


@Model.register("similar")
class SimilarBert(Model):

    def __init__(
        self,
        vocab: Vocabulary,
        label_namespace: str = "label",
        model_path: str = "bert",
        **kwargs
    ) -> None:
        super().__init__(vocab, **kwargs)

        self._text_field_embedder = PretrainedTransformerEmbedder(model_path)
        self._text_field_embedder = BasicTextFieldEmbedder(
            {"tokens": self._text_field_embedder})

        # 输出
        self._pooler = BertPooler(model_path, dropout=0.1)
        self._output_layer = torch.nn.Linear(self._text_field_embedder.get_output_dim()*2, 2)
        self._output_layer.weight.data.normal_(mean=0.0, std=0.02)
        self._output_layer.bias.data.zero_()

        # loss
        self._loss = torch.nn.CrossEntropyLoss()
        self._accuracy = CategoricalAccuracy()

        self._label_namespace = label_namespace

    def polled(self, embedding):
        cls = torch.index_select(embedding, dim=1,
                                 index=torch.tensor([0], device=embedding.device))
        cls = cls.squeeze()
        avg = torch.mean(embedding, dim=1)
        return torch.cat([cls, avg], dim=1)

    def forward(self,
                tokens: TextFieldTensors,
                label: Optional[torch.IntTensor] = None
                ) -> Dict[str, torch.Tensor]:

        embedding = self._text_field_embedder(tokens)
        pooled_embedding = self.polled(embedding)
        logits = self._output_layer(pooled_embedding)

        result = {"logits": logits}
        result["probs"] = torch.nn.functional.softmax(logits, dim=-1)
        if label is not None:
            result["loss"] = self._loss(logits, label)
            self._accuracy(logits, label)
            self.make_output_human_readable(result)

        return result

    @overrides
    def make_output_human_readable(
            self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        predictions = output_dict["probs"]
        predictions_list = [predictions[i] for i in range(predictions.shape[0])]

        classes = []
        label_probs = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_index_to_token_vocabulary(self._label_namespace)\
                .get(label_idx, str(label_idx))
            classes.append(label_idx)
            label_probs.append(prediction[label_idx])

        output_dict["label"] = classes
        output_dict["label_probs"] = label_probs
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, Dict[str, Any]]:
        return {
            "accuracy": self._accuracy.get_metric(reset),
        }


