"""run with pytorch-lightning"""
import argparse
import copy
from datetime import datetime
import time
import os
import math
from pprint import pprint

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import dill as pkl
import pickle
import pandas as pd
import wandb
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

DEVICE = 'cuda'

class InteractionDataset(torch.utils.data.Dataset):
    def __init__(self, uid2sequence, seq_len=100, stride=50, \
        last_max_seq_only=False, last_interaction_only=False):
        self.seq_len = seq_len
        if stride is None:
            self.stride = seq_len // 2
        else:
            self.stride = stride
        self.last_max_seq_only = last_max_seq_only  # If True, test last max-seq-len interaction only.
        self.last_interaction_only = last_interaction_only  # If True, test last single interaction only.
        # TODO: Resolve above two variables into a single test_mode variable.
        self.uid2sequence = uid2sequence
        self.sample_list = []
        for uid, seq in tqdm(self.uid2sequence.items()):
            num_inter = len(seq["item_id"])
            if self.last_max_seq_only:
                self.sample_list.append((uid, max(0, num_inter - self.seq_len), num_inter))
            else:
                start_idx, end_idx = 0, self.seq_len
                while end_idx < num_inter:
                    self.sample_list.append((uid, start_idx, end_idx))
                    start_idx += self.stride
                    end_idx += self.stride
                # Here, end_idx >= num_inter for the first time
                end_idx = num_inter
                self.sample_list.append((uid, start_idx, end_idx))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        uid, start_idx, end_idx = self.sample_list[index]
        fragment_len = end_idx - start_idx
        features = self.uid2sequence[uid]
        pad_size = self.seq_len - fragment_len
        pad = pad_size * [0]
        pad_mask = [True] * fragment_len + [False] * pad_size

        features = {
            "qid": [q + 1 for q in features["item_id"][start_idx:end_idx]] + pad,
            "is_correct": [2 - c for c in features["correct"][start_idx:end_idx]] + pad,
            "skill": [s + 1 for s in features["skill_id"][start_idx:end_idx]] + pad,
            "pad_mask": pad_mask,
        }
        if start_idx != 0:
            infer_mask = [False] * (self.seq_len - self.stride) + [True] * self.stride
        else:
            infer_mask = [True] * self.seq_len

        if self.last_interaction_only:
            infer_mask = [False] * (fragment_len - 1) + [True] + [False] * pad_size
        features["infer_mask"] = [(x and y) for (x, y) in zip(infer_mask, pad_mask)]

        return {
            "qid": torch.LongTensor(features["qid"]),
            "skill": torch.LongTensor(features["skill"]),
            "is_correct": torch.LongTensor(features["is_correct"]),
            "pad_mask": torch.BoolTensor(features["pad_mask"]),
            "infer_mask": torch.BoolTensor(features["infer_mask"]),
        }


def get_data(dataset, overwrite_test_df=None):
    data = {}
    modes = ["train", "val", "test"]
    if dataset in ["ednet", "ednet_medium"]:
        for mode in modes:
            with open(f"data/{dataset}/{mode}_data.pkl", "rb") as file:
                data[mode] = pkl.load(file)
    else:
        train_df = pd.read_csv(
            os.path.join("data", dataset, "preprocessed_data_train.csv"), sep="\t"
        )
        if overwrite_test_df is not None:
            test_df = overwrite_test_df
        else:
            test_df = pd.read_csv(
                os.path.join("data", dataset, "preprocessed_data_test.csv"), sep="\t"
            )

        data = {mode: {} for mode in modes}
        for (uid, _data) in tqdm(test_df.groupby("user_id")):
            seqs = _data.to_dict()
            del seqs["user_id"], seqs["timestamp"]
            data["test"][uid] = {key: list(x.values()) for key, x in seqs.items()}
        if overwrite_test_df is None:
            train_val = {}
            for (uid, _data) in tqdm(train_df.groupby("user_id")):
                seqs = _data.to_dict()
                del seqs["user_id"], seqs["timestamp"]
                train_val[uid] = {key: list(x.values()) for key, x in seqs.items()}
            num_val_users = len(train_val) // 8
            _train_users = list(train_val.keys())
            np.random.shuffle(_train_users)
            val_users = _train_users[:num_val_users]
            for uid, seq in train_val.items():
                if uid in val_users:
                    data["val"][uid] = seq
                else:
                    data["train"][uid] = seq
    return data


class DataModule(pl.LightningDataModule):
    def __init__(self, config, overwrite_test_df=None, last_one_only=False):
        super().__init__()
        self.data = get_data(config.dataset, overwrite_test_df=overwrite_test_df)
        if overwrite_test_df is None:
            train_data = InteractionDataset(self.data["train"], seq_len=config.seq_len,)
            val_data = InteractionDataset(self.data["val"], seq_len=config.seq_len, stride=10)
            self.train_gen = torch.utils.data.DataLoader(
                dataset=train_data,
                shuffle=True,
                batch_size=config.train_batch,
                num_workers=config.num_workers,
            )
            self.val_gen = torch.utils.data.DataLoader(
                dataset=val_data,
                shuffle=False,
                batch_size=config.test_batch,
                num_workers=config.num_workers,
            )
        else:
            self.train_gen = None
            self.val_gen = None

        test_data = InteractionDataset(
            self.data["test"], stride=1, seq_len=config.seq_len, \
                last_max_seq_only=False, last_interaction_only=last_one_only
        )
        self.test_gen = torch.utils.data.DataLoader(
            dataset=test_data,
            shuffle=False,
            batch_size=config.test_batch,
            num_workers=config.num_workers,
        )

    def train_dataloader(self):
        return self.train_gen

    def test_dataloader(self):
        return self.test_gen

    def val_dataloader(self):
        return self.val_gen


class AbsoluteDiscretePositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from the original transformer.
    """

    def __init__(self, dim_emb, max_len=5000, scale_const=10000.0, device="cpu"):
        super().__init__()
        if dim_emb % 2 != 0:
            raise ValueError("embedding dimension should be an even number")
        pos_enc = torch.zeros(max_len + 1, dim_emb).to(device)
        position = (
            torch.arange(0, max_len + 1, dtype=torch.float).unsqueeze(1).to(device)
        )
        div_term = torch.exp(
            torch.arange(0, dim_emb, 2).float() * (-math.log(scale_const) / dim_emb)
        ).to(device)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_enc", pos_enc)

    def forward(self, timestamps):
        """
        Args:
            timestamps: 1-D LongTensor of shape (L, ) which is [1, 2, ..., L]
        Returns:
            absolute positional encoding: 3-D FloatTensor of shape (1, L, D)
        """
        abs_pos_enc = self.pos_enc[timestamps].unsqueeze(0)  # (1, L, D)
        return abs_pos_enc


class SAINT(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embed = nn.ModuleDict()
        # maybe TODO: manage them with feature classes
        self.embed["qid"] = nn.Embedding(
            config.num_item + 1, config.dim_model, padding_idx=0
        )
        if hasattr(config, "num_part"):
            self.embed["part"] = nn.Embedding(
                config.num_part + 1, config.dim_model, padding_idx=0
            )
        self.embed["skill"] = nn.Embedding(
            config.num_skill + 1, config.dim_model, padding_idx=0
        )
        self.embed["is_correct"] = nn.Embedding(3, config.dim_model, padding_idx=0)

        # transformer
        self.transformer = nn.Transformer(
            d_model=config.dim_model,
            nhead=config.head_count,
            num_encoder_layers=config.layer_count,
            num_decoder_layers=config.layer_count,
            dim_feedforward=config.dim_ff,
            dropout=config.dropout_rate,
        )
        # positional encoding
        self.embed["enc_pos"] = AbsoluteDiscretePositionalEncoding(
            dim_emb=config.dim_model, max_len=config.seq_len, device=config.device
        )
        self.embed["dec_pos"] = copy.deepcopy(self.embed["enc_pos"])
        self.generator = nn.Linear(config.dim_model, 1)

        self.val_auc = 0
        self.best_val_auc = 0
        self.best_step = -1
        self.test_auc = 0

        # xavier initialization
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            
    def compute_all_losses(
        self, batch_dict,
    ):
        """
        forward function for Transformer
        Args:
            batch_dict: dictionary of float tensors.
            Used items:
                observed_data: observed features, (B, L_comb, f_num)
                observed_tp: combined timestamps (L_comb, ), same as
                             tp_to_predict
                observed_mask: 1 for observed features, 0 for otherwise
            Unused items:
                tp_to_predict: equal to tp_to_predict
                mode: stringnvidia
            n_tp_to_sample: not used
            n_traj_samples: not usedo
            kl_coeff: not used
        Returns:
            ...
        """
        device = self.config.device
        timestamp = torch.arange(1, self.config.seq_len + 1).to(device)

        obs_mask = batch_dict["pad_mask"]  # padding mask
        src = self.embed["qid"](batch_dict["qid"])
        src += self.embed["skill"](batch_dict["skill"].squeeze(-1))
        # src += self.embed['part'](batch_dict['part'])
        # positional encoding
        enc_pos = self.embed["enc_pos"](timestamp).squeeze(0)  # (L_T, D)
        src += enc_pos

        shifted_correct = torch.cat(
            [
                torch.zeros([batch_dict["is_correct"].size(0), 1]).long().to(device),
                batch_dict["is_correct"][:, :-1],
            ],
            -1,
        )
        tgt = self.embed["is_correct"](shifted_correct)
        dec_pos = self.embed["dec_pos"](timestamp).squeeze(0)  # (B, L_T, D)
        tgt += dec_pos

        src = src.transpose(0, 1)  # (L, B, D)
        tgt = tgt.transpose(0, 1)
        attn_mask = self.transformer.generate_square_subsequent_mask(src.size(0))
        attn_mask = attn_mask.to(device)

        transformer_output = self.transformer(
            src,
            tgt,
            src_mask=attn_mask,
            tgt_mask=attn_mask,
            memory_mask=attn_mask,
            src_key_padding_mask=~obs_mask,
            tgt_key_padding_mask=~obs_mask,
            # memory_key_padding_mask=obs_mask
        ).transpose(
            0, 1
        )  # (B, L, D)

        prediction = self.generator(transformer_output).squeeze(-1)  # (B, L)
        y = batch_dict["is_correct"].float()
        ce_loss = nn.BCEWithLogitsLoss(reduction="none")(prediction, 2 - y)  # (B, L)

        results = {
            "loss": ce_loss,
            "pred": prediction.detach(),
        }
        return results

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr,)

        def noam(step: int):
            step = max(1, step)
            warmup_steps = self.config.warmup_step
            scale = warmup_steps ** 0.5 * min(
                step ** (-0.5), step * warmup_steps ** (-1.5)
            )
            return scale

        scheduler = LambdaLR(optimizer=optimizer, lr_lambda=noam)
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]
        return optimizer

    def training_step(self, train_batch, batch_idx):
        train_res = self.compute_all_losses(train_batch)
        loss = train_res["loss"]  # (B, L)
        mask = train_batch["pad_mask"]
        loss = torch.sum(loss * mask) / torch.sum(mask)
        # if (
        #     self.config.use_wandb
        #     and self.global_step % self.config.val_check_steps == 0
        # ):
        #     # wandb log
        #     wandb.log({"Train loss": loss.item()}, step=self.global_step)
        return {"loss": loss}

    def training_epoch_end(self, training_step_outputs):
        # summarize train epoch results
        losses = []
        for out in training_step_outputs:
            losses.append(out["loss"])
        epoch_loss = sum(losses) / len(losses)
        print(f"Epoch Train loss: {epoch_loss}")

    def validation_step(self, val_batch, batch_idx):
        val_res = self.compute_all_losses(val_batch)
        loss = val_res["loss"]  # (B, L)
        pred = val_res["pred"]
        label = 2 - val_batch["is_correct"]

        infer_mask = val_batch["infer_mask"]
        loss = torch.sum(loss * infer_mask) / torch.sum(infer_mask)
        nonzeros = torch.nonzero(infer_mask, as_tuple=True)
        pred = pred[nonzeros].sigmoid()
        label = label[nonzeros].long()
        return {"loss": loss, "pred": pred, "label": label}

    def validation_epoch_end(self, validation_step_outputs):
        # summarize epoch results
        losses = []
        preds = []
        labels = []
        for out in validation_step_outputs:
            losses.append(out["loss"])
            preds.append(out["pred"])
            labels.append(out["label"])

        preds = torch.cat(preds, dim=0).view(-1)
        labels = torch.cat(labels, dim=0).view(-1)
        epoch_loss = sum(losses) / len(losses)
        epoch_auc = roc_auc_score(labels.cpu(), preds.cpu())

        print(
            f"Val loss: {epoch_loss:.4f}, auc: {epoch_auc:.4f}, previous best auc: {self.best_val_auc:.4f}"
        )
        if epoch_auc > self.best_val_auc:
            self.best_val_auc = epoch_auc
            self.best_step = self.global_step

        if self.config.use_wandb:
            wandb.log(
                {
                    "Val loss": epoch_loss,
                    "Val auc": epoch_auc,
                    "Best Val auc": self.best_val_auc,
                },
                step=self.global_step + 1,
            )

        return {
            "val_loss": epoch_loss,
            "val_auc": epoch_auc,
            "best_val_auc": self.best_val_auc,
            "best_step": self.best_step,
        }

    def test_step(self, test_batch, batch_idx):
        test_res = self.compute_all_losses(test_batch)
        loss = test_res["loss"]  # (B, L)
        pred = test_res["pred"]
        label = 2 - test_batch["is_correct"]

        infer_mask = test_batch["infer_mask"]
        loss = torch.sum(loss * infer_mask) / torch.sum(infer_mask)
        nonzeros = torch.nonzero(infer_mask, as_tuple=True)
        pred = pred[nonzeros].sigmoid()
        label = label[nonzeros].long()
        return {"pred": pred, "label": label, "loss": loss}

    def test_epoch_end(self, test_step_outputs):
        losses = []
        preds = []
        labels = []
        for out in test_step_outputs:
            losses.append(out["loss"])
            preds.append(out["pred"])
            labels.append(out["label"])

        preds = torch.cat(preds, dim=0).view(-1)
        labels = torch.cat(labels, dim=0).view(-1)
        epoch_loss = sum(losses) / len(losses)
        epoch_auc = roc_auc_score(labels.cpu(), preds.cpu())
        print(f"Test loss: {epoch_loss:.4f}, auc: {epoch_auc:.4f}")
        self.test_auc = epoch_auc
        if self.config.use_wandb:
            wandb.log(
                {"Test loss": epoch_loss, "Test auc": epoch_auc,}
            )

        return {
            "test_auc": epoch_auc
        }
        
    def _log_final_results(self):
        result_path = f"results/saint_{args.dataset}.csv"
        column_names = [
            "experiment_time",
            "train_batch",
            "val_check_interval",
            "lr",
            "warmup_step",
            "dropout_rate",
            "layer_count",
            "head_count",
            "dim_model",
            "dim_ff",
            "seq_len",
            "best_val_auc",
            "best_step",
            "test_auc",
        ]
        if not os.path.exists(result_path):
            # initialize result dataframe
            df = pd.DataFrame(columns=column_names)
            df.to_csv(result_path, index=False, header=True)

        base_df = pd.read_csv(result_path)
        current_time = datetime.now()
        result_df = pd.DataFrame.from_dict([{
            "experiment_time": current_time,
            "train_batch": self.config.train_batch,
            "val_check_interval": self.config.val_check_interval,
            "lr": self.config.lr,
            "warmup_step": self.config.warmup_step,
            "dropout_rate": self.config.dropout_rate,
            "layer_count": self.config.layer_count,
            "head_count": self.config.head_count,
            "dim_model": self.config.dim_model,
            "dim_ff": self.config.dim_ff,
            "seq_len": self.config.seq_len,
            "best_val_auc": self.best_val_auc,
            "best_step": self.best_step,
            "test_auc": epoch_auc,
        }])
        base_df = base_df.append(result_df)
        base_df.to_csv(result_path, index=False, header=True)


def str2bool(val):
    if val.lower() in ("yes", "true", "t", "y", "1"):
        ret = True
    elif val.lower() in ("no", "false", "f", "n", "0"):
        ret = False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    return ret


def predict_saint(saint_model, dataloader):
    preds = []
    for batch in tqdm(dataloader):
        test_res = saint_model.compute_all_losses(batch)
        pred = test_res["pred"]
        infer_mask = batch["infer_mask"]
        nonzeros = torch.nonzero(infer_mask, as_tuple=True)
        pred = pred[nonzeros].sigmoid()
        preds.append(pred)
    preds = torch.cat(preds, dim=0).view(-1)
    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--project", type=str, default='rebenchmark')
    parser.add_argument("--name", type=str, default='saint')
    parser.add_argument("--val_check_interval", type=float, default=1.0)
    parser.add_argument("--random_seed", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--train_batch", type=int, default=64)
    parser.add_argument("--test_batch", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--layer_count", type=int, default=2)
    parser.add_argument("--head_count", type=int, default=8)
    parser.add_argument("--warmup_step", type=int, default=4000)
    parser.add_argument("--dim_model", type=int, default=64)
    parser.add_argument("--dim_ff", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=150)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--dataset", type=str, default="ednet_small")
    # for debugging
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    parser.add_argument("--limit_test_batches", type=float, default=1.0)
    args = parser.parse_args()

    # parse gpus
    if args.gpu is not None:
        args.device = "cuda"
        args.gpu = [
            int(g) for g in args.gpu.split(",")
        ]  # doesn't support multi-gpu yet
    else:
        args.device = "cpu"

    if args.name is None:
        args.name = (
            f"{args.dataset}_l{args.layer_count}_dim{args.dim_model}_seq{args.seq_len}"
            + f"_{int(time.time())}"
        )

    if args.dataset in ["ednet", "ednet_medium"]:
        args.num_item = 14000
        args.num_skill = 300
    else:
        full_df = pd.read_csv(
            os.path.join("data", args.dataset, "preprocessed_data.csv"), sep="\t"
        )
        args.num_item = int(full_df["item_id"].max() + 1)
        args.num_skill = int(full_df["skill_id"].max() + 1)
    # set random seed
    pl.seed_everything(args.random_seed)

    model = SAINT(args)
    datamodule = DataModule(args)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_auc",
        dirpath=f"save/saint/{args.dataset}",
        filename="{val_auc:.4f}",
        mode="max",
    )
    early_stopping = EarlyStopping(
        monitor="val_auc",
        patience=5,
        mode="max",
    )
    trainer = pl.Trainer(
        gpus=args.gpu,
        accelerator='ddp',
        callbacks=[checkpoint_callback, early_stopping],
        max_epochs=args.num_epochs,
        val_check_interval=args.val_check_interval,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
    )
    # initialize wandb
    if args.use_wandb:
        wandb.init(project=args.project, name=args.name, config=args)
        print('wandb init')
    trainer.fit(model=model, datamodule=datamodule)

    # validation results
    best_val_auc = trainer.callback_metrics["best_val_auc"]
    best_step = trainer.callback_metrics["best_step"]

    # test results
    checkpoint_path = checkpoint_callback.best_model_path
    model = SAINT.load_from_checkpoint(checkpoint_path, config=args)
    with open(checkpoint_path.replace('.ckpt', '_config.pkl'), 'wb+') as file:
        pickle.dump(args.__dict__, file)
    trainer.test(model=model, datamodule=datamodule)
    test_auc = trainer.callback_metrics["test_auc"]

    # log results
    result_path = f"results/saint_{args.dataset}.csv"
    column_names = [
        "experiment_time",
        "train_batch",
        "val_check_interval",
        "lr",
        "warmup_step",
        "dropout_rate",
        "layer_count",
        "head_count",
        "dim_model",
        "dim_ff",
        "seq_len",
        "best_val_auc",
        "best_step",
        "test_auc",
    ]
    if not os.path.exists(result_path):
        # initialize result dataframe
        df = pd.DataFrame(columns=column_names)
        df.to_csv(result_path, index=False, header=True)

    base_df = pd.read_csv(result_path)
    current_time = datetime.now()
    result_df = pd.DataFrame.from_dict([{
        "experiment_time": current_time,
        "train_batch": args.train_batch,
        "val_check_interval": args.val_check_interval,
        "lr": args.lr,
        "warmup_step": args.warmup_step,
        "dropout_rate": args.dropout_rate,
        "layer_count": args.layer_count,
        "head_count": args.head_count,
        "dim_model": args.dim_model,
        "dim_ff": args.dim_ff,
        "seq_len": args.seq_len,
        "best_val_auc": best_val_auc,
        "best_step": best_step,
        "test_auc": test_auc,
    }])
    base_df = base_df.append(result_df)
    base_df.to_csv(result_path, index=False, header=True)
