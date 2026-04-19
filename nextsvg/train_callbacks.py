import lightning
import lightning.pytorch.utilities
import torch


class GradientNormLogger(lightning.Callback):
    """
    Logs the gradient norm.
    This should log the gradient norm unscaled and before clipping.
    """

    def __init__(self, norm_type: float | int | str = 2):
        super().__init__()
        self.norm_type = norm_type

    def on_before_optimizer_step(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        log_every_n_steps = getattr(trainer, "log_every_n_steps", None)
        assert isinstance(log_every_n_steps, int)
        if trainer.global_step % log_every_n_steps != 0:
            return
        norms = lightning.pytorch.utilities.grad_norm(pl_module, norm_type=self.norm_type)
        pl_module.log_dict(norms)
