import functools
import warnings
from typing import NamedTuple, Protocol, runtime_checkable

import lightning
import numpy as np
import PIL.Image
import svglab
import svglab.serialize
import torch
import torchvision
import torchvision.transforms.functional
import transformers
from tensordict import TensorDict
from torch import Tensor

import nextsvg.model
import nextsvg.optim


class Batch(NamedTuple):
    inputs: TensorDict | dict
    img: list[PIL.Image.Image]
    svg_str: list[str]

    def __len__(self):
        return len(self.svg_str)


@runtime_checkable
class ModelOutputs(Protocol):
    loss: Tensor


class VectorizerLightning(lightning.LightningModule):
    def __init__(
        self,
        architecture_config: dict,
        optimizer_config: dict,
        scheduler_config: dict,
        eval_generate_kwargs: dict,
        valid_ds_suffixes: list[str] | None = None,
        valid_ds_is_gen_eval: list[bool] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.eval_generate_kwargs = eval_generate_kwargs
        self.model: torch.nn.Module = nextsvg.model.build_model(architecture_config)  # TODO
        self.tokenizer: transformers.PreTrainedTokenizer | None = None
        self.valid_ds_suffixes = valid_ds_suffixes
        self.valid_ds_is_gen_eval = valid_ds_is_gen_eval

    def set_tokenizer(self, tokenizer: transformers.PreTrainedTokenizerFast) -> None:
        self.tokenizer = tokenizer

    def configure_optimizers(self):
        optimizer = nextsvg.optim.build_optimizer(self.model.parameters(), self.optimizer_config)
        if self.scheduler_config is None:
            return optimizer
        scheduler = transformers.get_scheduler(optimizer=optimizer, **self.scheduler_config)
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval="step"))

    def forward(self, inputs: TensorDict | dict) -> Tensor:
        return self.model(**inputs)

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        outputs = self(batch.inputs)
        assert isinstance(outputs, ModelOutputs)
        self.log("train/loss", outputs.loss.item(), batch_size=len(batch))
        return outputs.loss

    def validation_step(self, batch: Batch, batch_idx: int, dataloader_idx: int) -> None:
        if self.valid_ds_is_gen_eval and self.valid_ds_is_gen_eval[dataloader_idx]:
            metrics = self._validation_step_gen_eval(batch)
        else:
            metrics = self._validation_step_loss_eval(batch)
        suffix = self._get_ds_suffix(dataloader_idx)
        self.log_dict(
            {f"valid_{suffix}/{k}": v for k, v in metrics.items()}, batch_size=len(batch), add_dataloader_idx=False
        )

    def _validation_step_loss_eval(self, batch: Batch) -> dict[str, float]:
        output: ModelOutputs = self.model(**batch.inputs)  # type: ignore
        assert output.loss is not None
        return {"loss": output.loss.item()}

    def _validation_step_gen_eval(self, batch: Batch) -> dict[str, float]:
        # TODO make sure that preds do not contain input tokens
        # TODO log predictions and references to wandb for some examples
        assert self.tokenizer is not None, "Tokenizer must be provided for generative evaluation"
        preds_tokens: Tensor = self.model.generate(**batch.inputs, **self.eval_generate_kwargs)  # type: ignore
        preds_raw = self.tokenizer.decode(preds_tokens, skip_special_tokens=True)
        preds_svg = list(map(self.safe_parse, preds_raw))
        preds_img = list(map(self.render_pred, preds_svg, batch.img))
        trues_svg = list(map(self.safe_parse, batch.svg_str))
        if any(svg is None for svg in trues_svg):
            errs = [str_svg for str_svg, svg in zip(batch.svg_str, trues_svg) if svg is None]
            warnings.warn("These reference SVGs cannot be parsed inside validation step: ")
            for err in errs:
                warnings.warn(err)
        str_exact_match = [pred_raw == true_str for pred_raw, true_str in zip(preds_raw, batch.svg_str)]
        tok_contain_eos = (preds_tokens == self.tokenizer.eos_token_id).any(dim=-1)
        metrics = {
            **self.img_errors(preds_img, batch.img),
            "str_len_avg": np.mean([len(pred) for pred in preds_raw]).item(),
            "str_exact_match": np.mean(str_exact_match).item(),
            "str_parseable": np.mean([svg is not None for svg in preds_svg]).item(),
            "tok_contain_eos": tok_contain_eos.float().mean(),
        }
        return metrics

    def img_errors(self, preds: list[PIL.Image.Image], trues: list[PIL.Image.Image]) -> dict[str, float]:
        preds_tensors = [torchvision.transforms.functional.to_tensor(pred) for pred in preds]
        trues_tensors = [torchvision.transforms.functional.to_tensor(true) for true in trues]
        return {
            "img_mse": torch.stack(list(map(torch.nn.functional.mse_loss, preds_tensors, trues_tensors))).mean().item(),
            "img_mae": torch.stack(list(map(torch.nn.functional.l1_loss, preds_tensors, trues_tensors))).mean().item(),
        }

    def _get_ds_suffix(self, dataloader_idx: int) -> str:
        if self.valid_ds_suffixes:
            return self.valid_ds_suffixes[dataloader_idx]
        return f"valid_{dataloader_idx}"

    def safe_parse(self, svg_pred_str: str) -> svglab.Svg | None:
        try:
            return svglab.parse_svg(svg_pred_str)
        except Exception:
            return None

    def render_pred(self, pred_svg: svglab.Svg | None, label_img: PIL.Image.Image) -> PIL.Image.Image:
        if pred_svg is None:
            return self._err_img(label_img.size)
        try:
            return (
                pred_svg.render(width=label_img.width, background=svglab.Color("white"))
                .convert("RGB")
                .resize(label_img.size)
            )
        except Exception:
            return self._err_img(label_img.size)

    @functools.lru_cache(maxsize=32)
    def _err_img(self, size: tuple[int, int]) -> PIL.Image.Image:
        return PIL.Image.new("RGB", size, color="black")
