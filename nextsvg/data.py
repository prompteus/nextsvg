from typing import Any, Callable

import PIL.Image
import torch


class HFWrapper(torch.utils.data.Dataset):
    def __init__(
        self,
        hf_dataset: torch.utils.data.Dataset,
        image_col: str,
        label_col: str,
        preprocess: Callable[[PIL.Image.Image, str | None], Any],
        pass_label_to_preprocess: bool,
    ):
        self.hf_dataset = hf_dataset
        self.image_col = image_col
        self.label_col = label_col
        self.preprocess = preprocess
        self.pass_label_to_preprocess = pass_label_to_preprocess

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, index: int):
        image = self.hf_dataset[index][self.image_col]
        label_str = self.hf_dataset[index][self.label_col]
        preprocess_label = label_str if self.pass_label_to_preprocess else None
        inputs = self.preprocess(image, preprocess_label)
        return inputs, label_str
