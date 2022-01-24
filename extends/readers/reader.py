import json
from typing import Dict
from allennlp.common.file_utils import cached_path
from allennlp.common.util import JsonDict, List
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, LabelField, TextField, MetadataField, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from overrides import overrides


@DatasetReader.register("similar")
class SimilarReader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        tokenizer: Tokenizer = None,
        max_length: int = None,
    ) -> None:
        super().__init__(manual_distributed_sharding=True,
                         manual_multiprocess_sharding=True)
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        with open(cached_path(file_path), "r") as data_file:
            for line in self.shard_iterable(data_file):
                if not line:
                    continue

                record = json.loads(line)
                instance = self.text_to_instance(record)
                if instance is not None:
                    yield instance

    def _truncate_pair(self, tokens_a, tokens_b, max_length):
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    @overrides
    def text_to_instance(self, record: JsonDict) -> Instance:  # type: ignore
        fields: Dict[str, Field] = {}

        text1 = record["text1"]
        text2 = record["text2"]
        label = None
        if "label" in record:
            label = int(record["label"])

        # 构造输入
        text1_tokens = self._tokenizer.tokenize(text1)
        text2_tokens = self._tokenizer.tokenize(text2)
        if self._max_length is not None:
            self._truncate_pair(text1_tokens, text2_tokens, self._max_length)
        tokens = self._tokenizer.add_special_tokens(text1_tokens, text2_tokens)
        fields["tokens"] = TextField(tokens)

        # 构造 lable
        if label is not None:
            fields["label"] = LabelField(label, skip_indexing=True)
        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["tokens"]._token_indexers = self._token_indexers
