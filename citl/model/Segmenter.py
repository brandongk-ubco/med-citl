import pytorch_lightning as L
import torch
from torchmetrics.classification import JaccardIndex

from ..losses.FocalLoss import FocalLoss


class Segmenter(L.LightningModule):
    def __init__(
        self,
        model,
        num_classes,
        lr=1e-3,
        lr_method="plateau",
        loss_function="cross_entropy",
        margin_weighting=False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model

        self.num_classes = num_classes

        self.jaccard = JaccardIndex(
            task="multiclass",
            num_classes=num_classes,
            average="none",
            ignore_index=0,
            zero_division=1.0,
        )

        self.val_jaccard = JaccardIndex(
            task="multiclass",
            num_classes=num_classes,
            average="none",
            ignore_index=0,
            zero_division=1.0,
        )

        self.test_jaccard = JaccardIndex(
            task="multiclass",
            num_classes=num_classes,
            average="none",
            ignore_index=0,
            zero_division=1.0,
        )

        self.lr = lr
        self.lr_method = lr_method
        self.margin_weighting = margin_weighting

        if loss_function == "cross_entropy":
            self.loss_function = torch.nn.CrossEntropyLoss(
                reduction="none", ignore_index=0
            )
        elif loss_function == "focal":
            self.loss_function = FocalLoss(
                "multiclass", reduction="none", from_logits=True, ignore_index=0
            )
        else:
            raise ValueError("Loss function not implemented")

    def loss(self, y_hat, y):
        y_long = y.long()
        loss = self.loss_function(y_hat, y_long)
        if self.margin_weighting:
            softmax = torch.softmax(y_hat, dim=1)
            weights = torch.gather(softmax, dim=1, index=y_long.unsqueeze(1)).squeeze(1)
            loss = loss * (1 - weights)
        return loss.mean()

    def forward(self, x):
        if x.dim() == 2:
            y_hat = self.model(x.unsqueeze(0).unsqueeze(0))
        elif x.dim() == 3:
            y_hat = self.model(x.unsqueeze(0))
        elif x.dim() == 4:
            y_hat = self.model(x)
        else:
            raise ValueError("Input must be 2, 3 or 4 dimensional")

        if isinstance(y_hat, tuple):
            y_hat, _ = y_hat

        return y_hat

    def on_train_epoch_start(self) -> None:
        self.jaccard.reset()

    def training_step(self, batch, batch_idx):
        x, y, _ = batch

        # if self.current_epoch == 0:
        #     img, target = x[1, :, :, :], y[1]
        #     if img.ndim > 2:
        #         img = img.moveaxis(0, -1)
        #     img = img - img.min()
        #     img = img / img.max()
        #     fig = visualize_segmentation(
        #         img.detach().cpu(), self.num_classes, mask=target[1:].detach().cpu()
        #     )
        #     if type(self.trainer.logger) is TensorBoardLogger:
        #         self.logger.experiment.add_figure(
        #             "example_image", fig, self.global_step
        #         )
        #     elif type(self.trainer.logger) is NeptuneLogger:
        #         self.logger.experiment["training/example_image"].append(fig)
        #     plt.close()

        y_hat = self(x)
        loss = self.loss(y_hat, y)

        jaccard = self.jaccard(y_hat.argmax(dim=1).long(), y.long())
        self.log("jaccard", torch.mean(jaccard[1:]))
        self.log_dict(
            dict(
                zip(
                    [f"jaccard_{c}" for c in self.trainer.datamodule.classes[1:]],
                    jaccard[1:],
                )
            ),
            on_step=True,
            on_epoch=False,
        )

        self.log("loss", loss)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_jaccard.reset()

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)

        val_loss = self.loss(y_hat, y)

        self.val_jaccard.update(y_hat.argmax(dim=1).long(), y.long())
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
        )

    def on_validation_epoch_end(self):
        jaccard = self.val_jaccard.compute()
        self.log(
            "val_jaccard",
            torch.mean(jaccard[1:]),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log_dict(
            dict(
                zip(
                    [f"val_jaccard_{c}" for c in self.trainer.datamodule.classes[1:]],
                    jaccard[1:],
                )
            ),
            on_step=False,
            on_epoch=True,
        )

    def on_test_epoch_start(self) -> None:
        self.test_jaccard.reset()

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)

        test_loss = self.loss(y_hat, y)

        self.test_jaccard.update(y_hat.argmax(dim=1).long(), y.long())

        self.log(
            "test_loss",
            test_loss,
            on_step=False,
            on_epoch=True,
        )

    def on_test_epoch_end(self):
        jaccard = self.test_jaccard.compute()
        self.log("test_jaccard", torch.mean(jaccard[1:]))
        self.log_dict(
            dict(
                zip(
                    [f"test_jaccard_{c}" for c in self.trainer.datamodule.classes[1:]],
                    jaccard[1:],
                )
            )
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.lr * 0.1
        )

        scheduler = None

        if self.lr_method == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.2,
                patience=10,
                min_lr=1e-6,
            )
            interval = "epoch"

        if scheduler:
            return [optimizer], [
                {
                    "scheduler": scheduler,
                    "interval": interval,
                    "monitor": "val_loss",
                }
            ]

        return optimizer


__all__ = ["Segmenter"]
