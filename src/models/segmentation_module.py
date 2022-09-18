from typing import Any, List
import logging

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
import segmentation_models_pytorch as smp

logging.getLogger('PIL').setLevel(logging.WARNING)


class SegmentationLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        num_classes: int,
        ignore_index: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        # loss function
        self.criterion = smp.losses.DiceLoss(
            mode=smp.losses.MULTICLASS_MODE,
            from_logits=True,
            ignore_index=ignore_index
        )

        # metric objects for calculating and averaging IoU across batches
        self.train_iou = MeanMetric()
        self.val_iou = MeanMetric()
        self.test_iou = MeanMetric()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation IoU
        self.val_iou_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_iou_best doesn't store iou from these checks
        self.val_iou_best.reset()

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y.long())
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def get_iou(self, preds, targets):
        tp, fp, fn, tn = smp.metrics.get_stats(
            preds, targets, mode='multiclass',
            ignore_index=self.hparams.ignore_index, num_classes=self.hparams.num_classes)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        return iou_score

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.train_loss(loss)

        iou_score = self.get_iou(preds, targets)
        self.train_iou(iou_score)

        self.log("train/loss", self.train_loss,
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/iou", self.train_iou, on_step=False,
                 on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.val_loss(loss)

        iou_score = self.get_iou(preds, targets)
        self.val_iou(iou_score)

        self.log("val/loss", self.val_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("val/iou", self.val_iou, on_step=False,
                 on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        iou = self.val_iou.compute()  # get current val IoU
        self.val_iou_best(iou)  # update best so far val IoU
        # log `val_iou_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/iou_best", self.val_iou_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.test_loss(loss)

        iou_score = self.get_iou(preds, targets)
        self.test_iou(iou_score)

        self.log("test/loss", self.test_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("test/iou", self.test_iou, on_step=False,
                 on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        scheduler = self.hparams.scheduler(optimizer=optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils
    import segmentation_models_pytorch as smp

    import torch.nn.functional as F

    def dice(y_pred, y_true):
        y_pred = y_pred.log_softmax(dim=1).exp()
        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        y_true = y_true.view(bs, -1)
        y_pred = y_pred.view(bs, num_classes, -1)
        y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
        y_true = y_true.permute(0, 2, 1)  # N, C, H*W
        y_true = y_true.type_as(y_pred)

        print(y_pred.shape, y_true.shape, y_pred.dtype, y_true.dtype)

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(
        root / "configs" / "model" / "unet_resnet.yaml")
    model = hydra.utils.instantiate(cfg)
    x = torch.zeros(10, 3, 320, 320)
    target = torch.zeros(10, 320, 320, dtype=torch.long)
    logit_mask = model(x)
    # dice(logit_mask, target)
    # print("out", logit_mask.shape)
    loss = smp.losses.DiceLoss(
        mode=smp.losses.MULTICLASS_MODE, from_logits=True)
    print("loss", loss(logit_mask, target))
