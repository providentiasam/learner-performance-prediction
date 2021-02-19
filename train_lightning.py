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
from models.model_lightning import SAKT, SAINT, DKT

DEVICE = 'cuda'
FIX_SUBTWO = False

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
        for uid, seq in tqdm(self.uid2sequence.items(), desc='User-wise Seq Chunking'):
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
            "is_correct": [((2 - c) if not FIX_SUBTWO else c) for c in features["correct"][start_idx:end_idx]] + pad,
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
    if dataset in []:
        for mode in modes:
            with open(f"data/{dataset}/{mode}_data.pkl", "rb") as file:
                data[mode] = pkl.load(file)
    else:
        if overwrite_test_df is not None:
            test_df = overwrite_test_df
        else:
            test_df = pd.read_csv(
                os.path.join("data", dataset, "preprocessed_data_test.csv"), sep="\t"
            )
        data = {mode: {} for mode in modes}
        for (uid, _data) in tqdm(test_df.groupby("user_id"), desc='Prepare Test'):
            seqs = _data.to_dict()
            del seqs["user_id"], seqs["timestamp"]
            data["test"][uid] = {key: list(x.values()) for key, x in seqs.items()}

        if overwrite_test_df is None:
            train_df = pd.read_csv(
                os.path.join("data", dataset, "preprocessed_data_train.csv"), sep="\t")
            train_val = {}
            for (uid, _data) in tqdm(train_df.groupby("user_id"), desc='Prepare Train/Valid'):
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
    def __init__(self, config, overwrite_test_df=None, last_one_only=False, overwrite_test_batch=None):
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
            batch_size=config.test_batch \
                if overwrite_test_batch is None \
                else overwrite_test_batch,
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


def str2bool(val):
    if val.lower() in ("yes", "true", "t", "y", "1"):
        ret = True
    elif val.lower() in ("no", "false", "f", "n", "0"):
        ret = False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    return ret


def predict_saint(saint_model, dataloader, return_labels=False):
    preds = []
    labels = []
    for batch in tqdm(dataloader, desc='Batch Processing'):
        test_res = saint_model.compute_all_losses(batch)
        pred = test_res["pred"]
        infer_mask = batch["infer_mask"]
        nonzeros = torch.nonzero(infer_mask, as_tuple=True)
        label = (2 - batch["is_correct"]) if not FIX_SUBTWO else batch["is_correct"]
        labels.append(label[nonzeros])
        pred = pred[nonzeros].sigmoid()
        preds.append(pred)
    preds = torch.cat(preds, dim=0).view(-1)
    if not return_labels:
        return preds
    else:
        labels = torch.cat(labels, dim=0).view(-1)
        return preds, labels


def print_args(args):
    """Print CLI arguments in a pretty form"""
    print("=" * 10 + " Experiment arguments " + "=" * 10)
    for arg in vars(args):
        print(f"{arg}:\t\t{getattr(args, arg)}")
    print("=" * 42)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--project", type=str, default='bt_lightning2')
    parser.add_argument("--dataset", type=str, default="ednet")
    parser.add_argument("--model", type=str, default='dkt')
    parser.add_argument("--name", type=str)
    parser.add_argument("--val_check_interval", type=float, default=1.0)
    parser.add_argument("--random_seed", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--train_batch", type=int, default=512)
    parser.add_argument("--test_batch", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gpu", type=str, default="0,1,2,3")
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--layer_count", type=int, default=2)
    parser.add_argument("--head_count", type=int, default=10)
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--warmup_step", type=int, default=500)
    parser.add_argument("--dim_model", type=int, default=50)
    parser.add_argument("--dim_ff", type=int, default=400)
    parser.add_argument("--seq_len", type=int, default=1000)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--accel", type=str, default='dp')
    args = parser.parse_args()

    # parse gpus
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        args.device = "cuda"
        args.gpu = [
            int(g) for g in range(len(args.gpu.split(',')))
        ]  # doesn't support multi-gpu yet
    else:
        args.device = "cpu"

    if args.name is None:
        args.name = (
            f"{args.model}_{args.dataset}_l{args.layer_count}_d{args.dim_model}_seq{args.seq_len}"
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

    print_args(args)
    if args.model.lower().startswith('saint'):
        model = SAINT(args)
    elif args.model.lower().startswith('sakt'):
        model = SAKT(args)
    elif args.model.lower().startswith('dkt'):
        model = DKT(args)
    else:
        raise NotImplementedError

    checkpoint_callback = ModelCheckpoint(
        monitor="val_auc",
        dirpath=f"save/{args.model}/{args.dataset}",
        filename=f"{args.name}",
        mode="max",
    )
    early_stopping = EarlyStopping(
        monitor="val_auc",
        patience=args.patience,
        mode="max",
    )
    trainer = pl.Trainer(
        gpus=args.gpu,
        accelerator=args.accel,
        auto_select_gpus=True,
        callbacks=[checkpoint_callback, early_stopping],
        max_epochs=args.num_epochs,
        val_check_interval=args.val_check_interval
    )
    # initialize wandb

    while True:
        try:
            datamodule = DataModule(args)
            trainer.fit(model=model, datamodule=datamodule)
            break
        except RuntimeError as e:
            print(e)
            args.train_batch = args.train_batch // 2
            wandb.log({"Train Batch": args.train_batch},
                step=model.global_step + 1,
            )
            if args.train_batch < 20:
                assert False


    # validation results
    best_val_auc = trainer.callback_metrics["best_val_auc"]
    best_step = trainer.callback_metrics["best_step"]

    # test results
    checkpoint_path = checkpoint_callback.best_model_path
    if args.model == 'saint':
        model = SAINT.load_from_checkpoint(checkpoint_path, config=args).cuda()
    elif args.model == 'sakt':
        model = SAKT.load_from_checkpoint(checkpoint_path, config=args).cuda()
    with open(checkpoint_path.replace('.ckpt', '_config.pkl'), 'wb+') as file:
        pickle.dump(args.__dict__, file)
    model.eval()
    trainer.test(model=model, datamodule=datamodule)
    test_auc = trainer.callback_metrics["test_auc"]
    print(test_auc)
    preds, labels = predict_saint(saint_model=model, dataloader=datamodule.test_gen, return_labels=True)
    print(roc_auc_score(labels.cpu(), preds.cpu()))
