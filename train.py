import os
import argparse
import shutil
import json
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from models.UNet import UNetConfig, UNet
from models.MAE import MAEtoUNet, ViTMAEForPreTraining
from data.dataset import RSDataset
from utils.learn import get_cosine_schedule_with_warmup, get_result_dir
from utils.metrics import score

torch.set_float32_matmul_precision("medium")
pl.seed_everything(3407)


class SegmentNet(pl.LightningModule):

    def __init__(self, config, model_config, result_dir="./results"):
        super().__init__()
        self.config = OmegaConf.load(config)
        with open(model_config) as f:
            self.model_config = json.load(f)
        if self.model_config["name"] == "UNet":
            self.model:nn.Module = UNet(UNetConfig.from_dict(self.model_config))
        elif self.model_config["name"] == "MAEtoUNet":
            mae = ViTMAEForPreTraining.from_pretrained("./models/pretrained")
            decoder = UNet(UNetConfig.from_dict(self.model_config))
            self.model:nn.Module = MAEtoUNet(mae, decoder)
        else:
            raise TypeError("There's No such Model!")
        print(f"Model: {self.model_config['name']}")
        self.result_dir = result_dir

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        inputs = batch["inputs"]
        labels = batch["labels"]
        outputs = self.forward(inputs, labels)
        metrics = score(outputs.logits.detach().cpu().numpy(), outputs.labels.detach().cpu().numpy())
        self.log("train_loss", outputs.loss)
        self.log("train_accuracy", metrics["accuracy"])
        self.log("train_f1", metrics["f1"])
        return outputs.loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        inputs = batch["inputs"]
        labels = batch["labels"]
        outputs = self.forward(inputs, labels)
        metrics = score(outputs.logits.detach().cpu().numpy(), outputs.labels.detach().cpu().numpy())
        self.log("valid_loss", outputs.loss)
        self.log("valid_accuracy", metrics["accuracy"])
        self.log("valid_f1", metrics["f1"])
        return outputs.loss

    def train_dataloader(self):
        dataset = RSDataset("train", self.config.data.DB_dir)
        train_loader = DataLoader(dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers, shuffle=True)
        return train_loader

    def val_dataloader(self):
        dataset = RSDataset("valid", self.config.data.DB_dir)
        valid_loader = DataLoader(dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers, shuffle=False)
        return valid_loader

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), self.config.lr)
        warmup_step = int(self.config.num_training_steps / 100)
        print("Warmup step: ", warmup_step)
        schedule = {
            "scheduler": get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_step, num_training_steps=self.config.num_training_steps),
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [schedule]

    def configure_callbacks(self):
        lr_monitor = LearningRateMonitor(logging_interval="step")
        checkpoint_callback = ModelCheckpoint(dirpath=f"./{result_dir}/checkpoints", filename="{epoch}-{valid_loss:.8f}", monitor="valid_loss", mode="min", save_last=True, save_top_k=20, save_weights_only=False)
        earlystop_callback = EarlyStopping(monitor="valid_loss", patience=self.config.es_patience, mode="min")
        return [lr_monitor, checkpoint_callback, earlystop_callback]


def main(config, model_config, result_dir):

    model = SegmentNet(config, model_config, result_dir)
    config = OmegaConf.load(config)
    print("Model Architecture:")
    print(model)

    logger = TensorBoardLogger(save_dir="/".join(result_dir.split("/")[:-1]), name=result_dir.split("/")[-1])

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=config["devices"],
        logger=logger,
        log_every_n_steps=1,
        precision=config["precision"],
        max_steps=config["num_training_steps"],
        default_root_dir="./results"
    )

    if config["pretrained_model"]:
        trainer.fit(model, ckpt_path=config["pretrained_model"])
    else:
        trainer.fit(model)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="RS Segmentation")
    parser.add_argument("--config", type=str, default="./train.yaml")
    parser.add_argument("--model", type=str, default="MAEtoUNet")
    args = parser.parse_args()
    print("Args in experiment:")
    print(args)

    config = args.config
    model = args.model
    model_config = f"./models/configs/{args.model}.json"

    result_dir = get_result_dir()
    os.makedirs(result_dir, exist_ok=False)
    shutil.copy(config, f"./{result_dir}/config.yaml")
    shutil.copy(model_config, f"./{result_dir}/{args.model}.json")
    print("Save results to:", result_dir)

    main(config, model_config, result_dir)
