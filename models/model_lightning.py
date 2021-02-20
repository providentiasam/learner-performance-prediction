import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_sakt import (
    future_mask,
    clone,
    attention,
    relative_attention,
    MultiHeadedAttention,
)
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
import wandb
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
FIX_SUBTWO = False

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


class NonSelfAttentionLayer(torch.nn.TransformerEncoderLayer):
    # Modified Transformer Encoder Layer for SAKT
    # FFW was added on top of SAKT
    def __init__(
        self, d_model, nhead, dim_feedforward=128, dropout=0.1, activation="relu"
    ):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)

    def forward(self, srcs, src_mask, src_key_padding_mask=None):
        src_query, src_key, src_value = srcs
        src2 = self.self_attn(
            src_query,
            src_key,
            src_value,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        src = src_query + self.dropout1(src2)  # TODO: Test residual
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class LightningKT(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_nonoverlap = True
        if config.use_wandb:
            wandb.init(project=config.project, name=config.name, config=config)
        self.val_auc = 0
        self.best_val_auc = 0
        self.best_step = -1
        self.test_auc = 0
        self.epoch = 0
        self.preds = []
        self.labels = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr,)
        self.optimizer = optimizer
        if self.config.optimizer != 'noam':
            return [optimizer], []
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

        self.scheduler = scheduler
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        train_res = self.compute_all_losses(train_batch)
        loss = train_res["loss"]  # (B, L)
        mask = train_batch["pad_mask"]
        infer_mask = train_batch["infer_mask"]
        if self.train_nonoverlap:
            loss = torch.sum(loss * mask * infer_mask) / torch.sum(mask * infer_mask)
        else:
            loss = torch.sum(loss * mask) / torch.sum(mask)
        return loss

    def training_epoch_end(self, training_step_outputs):
        # summarize train epoch results
        losses = []
        for out in training_step_outputs:
            losses.append(out["loss"])
        epoch_loss = sum(losses) / len(losses)
        print(f"Epoch Train loss: {epoch_loss}")
        self.epoch += 1
        if self.config.use_wandb:
            wandb.log(
                {
                    "Train loss": epoch_loss,
                    "Learning rate": self.optimizer.param_groups[0]['lr'],
                    "Epoch": self.epoch
                },
                step=self.global_step + 1,
            )


    def validation_step(self, val_batch, batch_idx):
        val_res = self.compute_all_losses(val_batch)
        loss = val_res["loss"]  # (B, L)
        pred = val_res["pred"]
        label = (2 - val_batch["is_correct"]) if not FIX_SUBTWO else val_batch["is_correct"]

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
        epoch_loss = (sum(losses) / len(losses)).mean()
        epoch_auc = roc_auc_score(labels.cpu(), preds.cpu())
        print(
            f"Val loss: {epoch_loss.mean():.4f}, auc: {epoch_auc:.4f}, previous best auc: {self.best_val_auc:.4f}"
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
        label = (2 - test_batch["is_correct"]) if not FIX_SUBTWO else test_batch["is_correct"]

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
        self.preds.append(preds.cpu())
        self.labels.append(labels.cpu())
        epoch_loss = (sum(losses) / len(losses)).mean()
        try:
            epoch_auc = roc_auc_score(labels.cpu(), preds.cpu())
        except ValueError as e:
            print(e)
            epoch_auc = 0

        print(f"Test loss: {epoch_loss:.4f}, auc: {epoch_auc:.4f}")
        self.test_auc = epoch_auc
        if self.config.use_wandb:
            wandb.log(
                {"Test loss": epoch_loss, "Test auc": epoch_auc,}
            )

        return {
            "test_auc": epoch_auc
        }


class DKT(LightningKT):
    def __init__(self, config):
        super().__init__(config)
        self.embed = nn.ModuleDict()
        self.embed_sum = False  # Input to LSTM should be config.dim_model anyways
        self.embed_pos = False

        embed_dim = config.dim_model if self.embed_sum else config.dim_model // 2
        self.embed["qid"] = nn.Embedding(config.num_item + 1, embed_dim, padding_idx=0)
        self.embed["skill"] = nn.Embedding(config.num_skill + 1, embed_dim, padding_idx=0)
        self.embed["pos"] = AbsoluteDiscretePositionalEncoding(
            dim_emb=config.dim_model, max_len=config.seq_len, device=config.device
        )
        self.lstm = nn.LSTM(config.dim_model * 2, config.dim_model, config.layer_count, batch_first=True).cuda()

        self.pre_generator = nn.Linear(config.dim_model * 2, config.dim_model)
        self.generator = nn.Linear(config.dim_model, 1)
        self.dropout = nn.Dropout(p=config.dropout_rate)

    def compute_all_losses(
        self, batch_dict,
    ):

        device = self.config.device
        timestamp = torch.arange(1, self.config.seq_len + 1).to(device)

        query_item = self.embed["qid"](batch_dict["qid"].to(device))
        query_skill = self.embed["skill"](batch_dict["skill"].to(device))
        if self.embed_sum:
            query = query_item + query_skill
        else:
            query = torch.cat([query_item, query_skill], dim=-1)
        
        shifted_infos = {}
        for info in ['qid', 'skill', 'is_correct']:
            shifted_infos[info] = torch.cat(
                [torch.zeros([batch_dict[info].size(0), 1]).long().to(device),
                batch_dict[info][:, :-1].to(device)], -1)
        
        input_item = self.embed["qid"](shifted_infos['qid'])
        input_skill = self.embed["skill"](shifted_infos['skill'])
        input_pos = self.embed["pos"](timestamp - 1).squeeze(0)

        if self.embed_sum:
            input = input_item + input_skill
        else:
            input = torch.cat([input_item, input_skill], dim=-1)

        if self.embed_pos:
            input = input + input_pos
        input = torch.cat([input, input], dim=-1)
        input[..., : self.config.dim_model] *= (1 - (shifted_infos["is_correct"] != 1).int()).unsqueeze(-1)
        input[..., self.config.dim_model:] *= (1 - (shifted_infos["is_correct"] != 2).int()).unsqueeze(-1)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(input)
        x = self.pre_generator(torch.cat([self.dropout(x), query], dim=-1))
        x = self.generator(torch.relu(self.dropout(x))).squeeze(-1)
        prediction = x
        y = batch_dict["is_correct"].float().to(device)
        ce_loss = nn.BCEWithLogitsLoss(reduction="none")(prediction.to(device), (2 - y))  # (B, L)

        results = {
            "loss": ce_loss,
            "pred": prediction.detach(),
        }
        return results



class SAINT(LightningKT):
    def __init__(self, config):
        super().__init__(config)

        self.embed = nn.ModuleDict()
        # maybe TODO: manage them with feature classes
        self.embed["qid"] = nn.Embedding(
            config.num_item + 1, config.dim_model, padding_idx=0
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

        # xavier initialization
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            
    def compute_all_losses(
        self, batch_dict,
    ):
        device = self.config.device
        timestamp = torch.arange(1, self.config.seq_len + 1).to(device)

        obs_mask = batch_dict["pad_mask"].to(device)  # padding mask
        src = self.embed["qid"](batch_dict["qid"].to(device))
        src += self.embed["skill"](batch_dict["skill"].squeeze(-1).to(device))
        enc_pos = self.embed["enc_pos"](timestamp).squeeze(0)  # (L_T, D)
        src += enc_pos

        shifted_correct = torch.cat(
            [
                torch.zeros([batch_dict["is_correct"].size(0), 1]).long().to(device),
                batch_dict["is_correct"][:, :-1].to(device),
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
        y = batch_dict["is_correct"].float().to(device)
        ce_loss = nn.BCEWithLogitsLoss(reduction="none")(prediction.to(device), (2 - y) if not FIX_SUBTWO else y)  # (B, L)
        results = {
            "loss": ce_loss,
            "pred": prediction.detach(),
        }
        return results



class SAKT(LightningKT):
    def __init__(self, config):
        super().__init__(config)
        self.embed = nn.ModuleDict()
        # maybe TODO: manage them with feature classes
        self.embed["qid"] = nn.Embedding(
            config.num_item + 1, config.dim_model, padding_idx=0
        )
        self.embed["skill"] = nn.Embedding(
            config.num_skill + 1, config.dim_model, padding_idx=0
        )
        self.lin_in = nn.Linear(config.dim_model * 2, config.dim_model)
        self.embed["is_correct"] = nn.Embedding(3, config.dim_model, padding_idx=0)

        # transformer
        self.encoder_layers = clone(NonSelfAttentionLayer(
            d_model=config.dim_model,
            nhead=config.head_count,
            dim_feedforward=config.dim_ff,
            dropout=config.dropout_rate,
        ), config.layer_count)
        self.layer_norms = clone(nn.LayerNorm(config.dim_model), config.layer_count)
        # positional encoding
        self.embed["enc_pos"] = AbsoluteDiscretePositionalEncoding(
            dim_emb=config.dim_model, max_len=config.seq_len, device=config.device
        )
        self.generator = nn.Linear(config.dim_model, 1)

        # xavier initialization
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def compute_all_losses(
        self, batch_dict,
    ):

        device = self.config.device
        timestamp = torch.arange(1, self.config.seq_len + 1).to(device)

        obs_mask = batch_dict["pad_mask"].to(device)  # padding mask
        query = self.embed["qid"](batch_dict["qid"].to(device))
        query += self.embed["skill"](batch_dict["skill"].squeeze(-1).to(device))
        query += self.embed["enc_pos"](timestamp).squeeze(0)  # (L_T, D)
        
        shifted_infos = {}
        for info in ['qid', 'skill', 'is_correct']:
            shifted_infos[info] = torch.cat(
                [torch.zeros([batch_dict[info].size(0), 1]).long().to(device),
                batch_dict[info][:, :-1].to(device)], -1)
        keyval = self.embed["qid"](shifted_infos['qid'])
        keyval += self.embed["skill"](shifted_infos['skill'].squeeze(-1).to(device))
        keyval += self.embed["enc_pos"](timestamp - 1).squeeze(0)
        keyval = torch.cat([keyval, keyval], dim=-1)
        keyval[..., : self.config.dim_model] *= (1 - (shifted_infos["is_correct"] != 1).int()).unsqueeze(-1)
        keyval[..., self.config.dim_model:] *= (1 - (shifted_infos["is_correct"] != 2).int()).unsqueeze(-1)
        keyval = self.lin_in(keyval) # B L D

        query = query.transpose(0, 1)  # (L, B, D)
        keyval = keyval.transpose(0, 1)
        attn_mask = self.generate_square_subsequent_mask(query.size(0))
        attn_mask = attn_mask.to(device)
        for i, encoder in enumerate(self.encoder_layers):
            keyval = encoder(
                (query, keyval, keyval), attn_mask, ~obs_mask
            )
            keyval = self.layer_norms[i](keyval)
        output = keyval.transpose(0, 1)

        prediction = self.generator(output).squeeze(-1)  # (B, L)
        y = batch_dict["is_correct"].float().to(device)
        ce_loss = nn.BCEWithLogitsLoss(reduction="none")(prediction.to(device), (2 - y))  # (B, L)

        results = {
            "loss": ce_loss,
            "pred": prediction.detach(),
        }
        return results



