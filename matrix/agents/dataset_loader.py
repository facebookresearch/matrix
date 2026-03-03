# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import abc
import logging
import os
from typing import Any, Dict, Generator, Optional

from datasets import load_dataset

logger = logging.getLogger(__name__)


class BaseDatasetLoader(abc.ABC):
    @abc.abstractmethod
    def load_data(self) -> Generator[Dict[str, Any], None, None]:
        """Load data from the dataset"""
        pass

    @abc.abstractmethod
    def total_count(self) -> Optional[int]:
        pass


class HuggingfaceDatasetLoader(BaseDatasetLoader):
    def __init__(
        self,
        name: str,
        split: str,
        cut_off: Optional[int] = None,
        data_files: str | None = None,
        hub_download: bool = False,  # useful when pyarrow can't handle complex json
    ):
        super().__init__()
        self.dataset_name = name
        self.split = split
        if isinstance(data_files, str):
            data_files = os.path.expanduser(data_files)
        elif isinstance(data_files, list):
            data_files = [os.path.expanduser(f) for f in data_files]
        self.data_files = data_files
        self.cut_off = cut_off
        self.hub_download = hub_download
        self._count = 0
        self.done = False

    def load_data(self) -> Generator[Dict[str, Any], None, None]:
        if self.data_files is not None and self.hub_download:
            import json

            from huggingface_hub import hf_hub_download

            # Download the JSON file, automatically cached locally
            json_path = hf_hub_download(
                repo_id=self.dataset_name,
                filename=self.data_files,
                repo_type="dataset",
            )

            # Load it with Python json to get nested dictionary/list
            with open(json_path, "r") as f:
                dataset_val = json.load(f)
        else:
            dataset_val = load_dataset(
                self.dataset_name,
                split=self.split,
                data_files=self.data_files,
                streaming=True,
            )
        self._count = 0

        for item in dataset_val:
            if self.cut_off is not None and self._count >= self.cut_off:
                break
            item = self.transform(item)
            self._count += 1
            yield item

        self.done = True

    def transform(self, item) -> Dict[str, Any]:
        return dict(item)

    def total_count(self) -> Optional[int]:
        return self._count if self.done else None
