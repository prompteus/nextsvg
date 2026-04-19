import os
import pathlib

import datasets
import hydra
import lightning
import lightning.pytorch.callbacks
import lightning.pytorch.loggers
import omegaconf
import pydantic
import torch
import transformers
import typer
import wandb
import wandb.sdk.wandb_run
import yaml

import nextsvg.data
import nextsvg.model
import nextsvg.train_callbacks
import nextsvg.trainer


class TorchConfig(pydantic.BaseModel, extra="forbid"):
    num_threads: int
    float32_matmul_precision: str


class DatasetConfig(pydantic.BaseModel, extra="forbid"):
    load_args: dict
    image_col: str
    label_col: str


class Config(pydantic.BaseModel, extra="forbid"):
    global_seed: int = 42
    torch: TorchConfig
    architecture: nextsvg.model.ArchitectureConfig
    scheduler: dict
    optimizer: dict
    trainer: dict
    tokenizer: str
    train_dataset: DatasetConfig
    valid_dataset: DatasetConfig
    train_loader: dict
    valid_loader: dict
    checkpointing: list[dict]
    eval_generate_kwargs: dict


def resolve_config(
    config_path: pathlib.Path,
    override: list[str] | None = None,
) -> tuple[omegaconf.DictConfig, Config]:
    with hydra.initialize_config_dir(str(config_path.parent.absolute()), version_base=None):
        omega_config = hydra.compose(config_name=config_path.stem, overrides=override)

    config_dict = omegaconf.OmegaConf.to_container(omega_config, resolve=True)
    config: Config = Config.model_validate(config_dict)
    return omega_config, config


app = typer.Typer(
    pretty_exceptions_show_locals=False,
    rich_markup_mode="rich",
)


@app.command()
def main(
    config_path: pathlib.Path = typer.Argument(..., help="Path to the training config file."),
    override: list[str] | None = typer.Option(
        None,
        "--override",
        help="Hydra config overrides to apply on top of the config file. Can be used multiple times.",
    ),
) -> None:
    omega_cfg, cfg = resolve_config(config_path, override)
    torch.set_float32_matmul_precision(cfg.torch.float32_matmul_precision)
    torch.set_num_threads(cfg.torch.num_threads)
    lightning.seed_everything(cfg.global_seed)

    typer.secho("Loading tokenizer...", fg=typer.colors.CYAN)
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.tokenizer)
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

    typer.secho("Loading dataset...", fg=typer.colors.CYAN)
    ds_train = nextsvg.data.HFWrapper(
        datasets.load_dataset(**cfg.train_dataset.load_args),
        image_col=cfg.train_dataset.image_col,
        label_col=cfg.train_dataset.label_col,
    )
    ds_valid = nextsvg.data.HFWrapper(
        datasets.load_dataset(**cfg.valid_dataset.load_args),
        image_col=cfg.valid_dataset.image_col,
        label_col=cfg.valid_dataset.label_col,
    )

    def collate_batch(examples: list[tuple[dict, str]]) -> nextsvg.trainer.Batch:
        inputs = ...  # TODO
        labels_str = [label_str for model_inputs, label_str in examples]
        return nextsvg.trainer.Batch(inputs, labels_str)

    typer.secho("Building dataloader...", fg=typer.colors.CYAN)
    train_loader = torch.utils.data.DataLoader(ds_train, collate_fn=collate_batch, **cfg.train_loader)
    valid_loader = torch.utils.data.DataLoader(ds_valid, collate_fn=collate_batch, **cfg.valid_loader)

    typer.secho("Building model...", fg=typer.colors.CYAN)
    lmodule = nextsvg.trainer.VectorizerLightning(
        architecture_config=cfg.architecture.model_dump(),  # TODO
        optimizer_config=cfg.optimizer,
        scheduler_config=cfg.scheduler,
        eval_generate_kwargs=cfg.eval_generate_kwargs,
    )
    lmodule.set_tokenizer(tokenizer)

    typer.secho("Setting up logging and checkpointing...", fg=typer.colors.CYAN)
    wandb_logger = lightning.pytorch.loggers.WandbLogger(project="music-ocr", save_dir=".wandb/")
    run: wandb.sdk.wandb_run.Run = wandb_logger.experiment
    output_dir = pathlib.Path(f"./checkpoints/{run.project}/{run.name}/")
    os.makedirs(output_dir, exist_ok=True)

    checkpointings = []
    for checkpoint_cfg in cfg.checkpointing:
        kwargs = {"dirpath": output_dir, **checkpoint_cfg}
        checkpointings.append(lightning.pytorch.callbacks.ModelCheckpoint(**kwargs))

    gradnorm_logger = nextsvg.train_callbacks.GradientNormLogger(norm_type=2)
    lr_logger = lightning.pytorch.callbacks.LearningRateMonitor(
        logging_interval="step", log_momentum=True, log_weight_decay=True
    )

    typer.secho("Building trainer...", fg=typer.colors.CYAN)
    trainer = lightning.Trainer(
        **cfg.trainer,
        logger=wandb_logger,
        callbacks=[gradnorm_logger, lr_logger] + checkpointings,
    )

    typer.secho("Saving preprocessor to checkpoint directory...", fg=typer.colors.CYAN)
    ...  # TODO?

    typer.secho("Saving config...", fg=typer.colors.CYAN)
    config_dict = cfg.model_dump(mode="json")
    config_yaml_resolved = yaml.dump(config_dict, sort_keys=False)
    config_yaml_orig = omegaconf.OmegaConf.to_yaml(omega_cfg)

    with open(output_dir / "config.yaml", "w") as f:
        f.write(config_yaml_orig)
    with open(output_dir / "config_resolved.yaml", "w") as f:
        f.write(config_yaml_resolved)

    if hasattr(run.config, "update"):
        # without this check, wandb.config.update crashes when multiple GPUs are visible:
        # see: https://github.com/Lightning-AI/pytorch-lightning/discussions/13157
        run.config.update(config_dict)

    wandb_logger.log_table(
        "config",
        columns=["config_yaml", "config_yaml_resolved"],
        data=[[config_yaml_orig, config_yaml_resolved]],
    )

    typer.secho("Training...", fg=typer.colors.CYAN)
    trainer.fit(lmodule, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    typer.secho("Exiting", fg=typer.colors.CYAN)


if __name__ == "__main__":
    app()
