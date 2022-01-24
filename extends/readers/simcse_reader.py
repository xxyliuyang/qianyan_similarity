import json
from typing import Dict
from allennlp.common.file_utils import cached_path
from allennlp.common.util import JsonDict
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from overrides import overrides


@DatasetReader.register("simcse")
class SimcseReader(DatasetReader):
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

    @overrides
    def text_to_instance(self, record: JsonDict) -> Instance:  # type: ignore
        fields: Dict[str, Field] = {}

        for i in range(1, 4):
            key = "text" + str(i)
            text = record[key]
            tokens = self._tokenizer.tokenize(text)
            if self._max_length is not None and self._max_length < len(tokens):
                tokens = tokens[:self._max_length] + [tokens[-1]]
            fields[key] = TextField(tokens, token_indexers=self._token_indexers)
        return Instance(fields)
