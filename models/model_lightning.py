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


import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.model_ctf import CompressiveTransformer
from sklearn.metrics import roc_auc_score


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
        self.train_preds, self.train_labels, self.train_losses = [], [], []

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
        pred = train_res["pred"]
        label = (2 - train_batch["is_correct"]) if not FIX_SUBTWO else train_batch["is_correct"]
        mask = train_batch["pad_mask"]
        infer_mask = train_batch["infer_mask"]
        nonzeros = torch.nonzero(infer_mask, as_tuple=True)
        pred = pred[nonzeros].sigmoid()
        label = label[nonzeros].long()
        if self.train_nonoverlap:
            loss = torch.sum(loss * mask * infer_mask) / torch.sum(mask * infer_mask)
        else:
            loss = torch.sum(loss * mask) / torch.sum(mask)
        self.train_preds.append(pred.detach().cpu())
        self.train_labels.append(label.detach().cpu())
        self.train_losses.append(loss.detach().cpu())
        return loss

    def training_epoch_end(self, training_step_outputs):
        # summarize train epoch results
        epoch_loss = sum(self.train_losses) / len(self.train_losses)
        preds = torch.cat(self.train_preds, dim=0).view(-1)
        labels = torch.cat(self.train_labels, dim=0).view(-1)
        epoch_auc = roc_auc_score(labels.cpu(), preds.cpu())
        print(f"Epoch Train loss: {epoch_loss}")
        self.epoch += 1
        if self.config.use_wandb:
            wandb.log(
                {
                    "Train loss": epoch_loss,
                    "Train auc": epoch_auc,
                    "Learning rate": self.optimizer.param_groups[0]['lr'],
                    "Epoch": self.epoch
                },
                step=self.global_step + 1,
            )
        self.train_losses = []
        self.train_preds = []
        self.train_labels = []


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
        self.embed_sum = False  # config.embed_sum  # Input to LSTM should be config.dim_model anyways
        self.embed_pos = False  # config.embed_pos
        if self.embed_pos:
            raise NotImplementedError

        embed_dim = config.dim_model if self.embed_sum else config.dim_model // 2
        self.embed_dim = embed_dim
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
        # TODO: sqrt scale embedding output?
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
        self.embed_dim = config.dim_model
        try:
            self.embed_pos = config.embed_pos
            self.embed_sum = config.embed_sum
        except Exception as e:
            self.embed_pos = True
            self.embed_sum = True

        embed_dim = config.dim_model if self.embed_sum else config.dim_model // 2
        self.embed_dim = embed_dim
        self.embed["qid"] = nn.Embedding(
            config.num_item + 1, self.embed_dim, padding_idx=0
        )
        self.embed["skill"] = nn.Embedding(
            config.num_skill + 1, self.embed_dim, padding_idx=0
        )
        self.lin_in = nn.Linear(config.dim_model * 2, config.dim_model)
        self.embed["is_correct"] = nn.Embedding(3, config.dim_model, padding_idx=0)

        self.encoder_layers = clone(NonSelfAttentionLayer(
            d_model=config.dim_model,
            nhead=config.head_count,
            dim_feedforward=config.dim_ff,
            dropout=config.dropout_rate,
        ), config.layer_count)
        self.layer_norms = clone(nn.LayerNorm(config.dim_model), config.layer_count)

        if self.embed_pos:
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
        query_q = self.embed["qid"](batch_dict["qid"].to(device))
        query_s = self.embed["skill"](batch_dict["skill"].to(device))
        
        if self.embed_sum:
            query = query_q + query_s
        else:
            query = torch.cat([query_q, query_s], dim=-1)
        if self.embed_pos: 
            query += self.embed["enc_pos"](timestamp).squeeze(0)  # (L_T, D)
        
        shifted_infos = {}
        for info in ['qid', 'skill', 'is_correct']:
            shifted_infos[info] = torch.cat(
                [torch.zeros([batch_dict[info].size(0), 1]).long().to(device),
                batch_dict[info][:, :-1].to(device)], -1)

        embed_q = self.embed["qid"](shifted_infos['qid'].to(device))
        embed_s = self.embed["skill"](shifted_infos['skill'].to(device))
        if self.embed_sum:
            keyval = embed_q + embed_s
        else:
            keyval = torch.cat([embed_q, embed_s], dim=-1)
        if self.embed_pos:
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


NUM_ITEM = 20000
NUM_PART = 7
MIN_SEQ_LEN = 10
MAX_ELAPSED_TIME = 300.0
MAX_LAG_TIME = 86400.0
MIN_SCORE = 250
START_TOKEN_BOOL = 3
START_TOKEN_QID = 0
START_TOKEN_PART = 0
START_TOKEN_CONT = 0.0
BOOL_TOKEN_TO_FLOAT = torch.Tensor([0.0, 1.0, 0.0, 0.0])
PADDING_TOKEN = 0

# class CompressiveKTConfig:
#     """
#     Configuration class used for constructing a compressive transformer
#     """
#
#     def __init__(
#         self,
#         layer_count: int = 4,
#         head_count: int = 8,
#         dim_embed: int = 128,
#         dim_model: int = 512,
#         dim_feedforward: int = 1024,
#         dim_score_gen: int = 512,
#         seq_len: int = 128,
#         memory_layers=None,
#         mem_len: int = 256,
#         cmem_len: int = None,
#         cmem_ratio: int = 4,
#         gru_gated_residual: bool = True,
#         mogrify_gru: bool = False,
#         attn_dropout: float = 0.1,
#         ff_glu: bool = False,
#         ff_dropout: float = 0.1,
#         attn_layer_dropout: float = 0.1,
#         reconstruction_attn_dropout: float = 0.0,
#         reconstruction_loss_weight: float = 1.0,
#         one_kv_head: bool = False,
#         val_check_steps: float = 0.1,
#         lr: float = 1e-3,
#         use_wandb: bool = False,
#     ):
#         self.seq_len = seq_len
#         self.mem_len = mem_len
#         self.cmem_len = cmem_len
#         self.cmem_ratio = cmem_ratio
#         self.memory_layers = memory_layers
#         self.layer_count = layer_count
#         self.head_count = head_count
#         self.dim_embed = dim_embed
#         self.dim_model = dim_model
#         self.dim_feedforward = dim_feedforward
#         self.dim_score_gen = dim_score_gen
#         self.gru_gated_residual = gru_gated_residual
#         self.mogrify_gru = mogrify_gru
#         self.attn_dropout = attn_dropout
#         self.ff_glu = ff_glu
#         self.ff_dropout = ff_dropout
#         self.attn_layer_dropout = attn_layer_dropout
#         self.reconstruction_attn_dropout = reconstruction_attn_dropout
#         self.reconstruction_loss_weight = reconstruction_loss_weight
#         self.one_kv_head = one_kv_head
#         self.val_check_steps = val_check_steps
#         self.lr = lr
#         self.use_wandb = use_wandb
#

class InteractionEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.ModuleDict(
            {
                "qid": nn.Embedding(config.num_item + 2, config.emb_dim, padding_idx=0),
                "skill": nn.Embedding(config.num_skill + 2, config.emb_dim, padding_idx=0),
                "is_correct": nn.Embedding(4, config.emb_dim, padding_idx=0),
                # 'is_on_time': nn.Embedding(4, config.emb_dim, padding_idx=0),
                # "elapsed_time": nn.Linear(1, config.emb_dim, bias=False),
            }
        )

    def forward(self, token_dict):
        output = 0.0
        for key, embed in self.embedding.items():
            output += embed(token_dict[key])
        return output


class CompressiveKT(LightningKT):
    """
    pytorch lightning module for compressive KT model
    """

    def __init__(self, config):
        super().__init__(config)
        self.model = CompressiveTransformer(config)
        self.embed = InteractionEmbedding(config)
        self.seq_len = config.seq_len
        self.dim_model = config.dim_model
        self.dim_embed = config.dim_model
        self.generator = nn.Linear(self.dim_model, 1)
        self.lr = config.lr
        self.use_wandb = config.use_wandb
        self.best_val_auc = 0

    def compute_all_losses(self, batch):
        infer_result = self.inference(batch)
        label = infer_result["label"]  # (B, L)
        logit = infer_result["logit"]  # (B, L)
        aux_loss = infer_result["aux_loss"]  # (0)

        bce_loss = nn.BCEWithLogitsLoss(reduction="none")(logit, label)
        return {"loss": bce_loss, "aux_loss": aux_loss, "pred": logit.detach()}

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def compute_roc_auc(self, label, pred):
        return roc_auc_score(label.cpu(), pred.cpu())

    def shift_feature(self, feature, start_token):
        """
        input
            feature: (B, L)
            batch_size = B
        output
            shifted_feature: (B, L)
        """
        shifted_feature = torch.cat(
            [
                torch.tensor(start_token).repeat([feature.size(0), 1]).type_as(feature),
                feature[:, :-1],
            ],
            dim=-1,
        )  # (B, L)
        return shifted_feature

    def pass_to_model(self, input_dict, mask):
        """
        input
            input_dict: qid, part, is_correct, elapsed_time: (B, L)
            mask: (B, L)
        output
            output: (B, L, D)
            aux_loss: dim 0
        """
        x = self.embed(input_dict)  # (B, L, D)

        x = torch.unbind(
            x.view(x.size(0), -1, self.seq_len, x.size(-1)).transpose(
                0, 1
            )  # (n, B, l, D)
        )  # list of n tensors of size (B, l, D)
        mask = torch.unbind(
            mask.view(mask.size(0), -1, self.seq_len).transpose(0, 1)
        )  # list of n tensors of size (B, l)

        mem = None
        aux_losses = []
        outputs = []
        for _x, _mask in zip(x, mask):
            output, mem, aux_loss = self.model(_x, memories=mem, mask=_mask)
            outputs.append(output)  # output: (B, l, D)
            aux_losses.append(aux_loss)
        output = torch.cat(outputs, dim=1)  # (B, L, D)
        aux_loss = sum(aux_losses) / len(aux_losses)

        return output, aux_loss

    def inference(self, feature_dict):
        pad_mask = feature_dict["pad_mask"]  # (B, L)
        qid = feature_dict["qid"]  # (B, L)
        part = feature_dict["skill"]  # (B, L)
        is_correct = self.shift_feature(
            feature_dict["is_correct"], START_TOKEN_BOOL
        )  # (B, L)
        # elapsed_time = self.shift_feature(
        #     feature_dict["elapsed_time"], START_TOKEN_CONT
        # )  # (B, L)

        input_dict = {
            "qid": qid,
            "skill": part,
            "is_correct": is_correct,
            # "elapsed_time": elapsed_time.unsqueeze(-1),
        }
        output, aux_loss = self.pass_to_model(input_dict, pad_mask)

        logit = self.generator(output).squeeze(-1)  # (B, L)
        label = BOOL_TOKEN_TO_FLOAT[feature_dict["is_correct"]].type_as(feature_dict["is_correct"])

        return {"label": label, "logit": logit, "aux_loss": aux_loss}

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    #     return {"optimizer": optimizer}

    def training_step(self, train_batch, batch_idx):
        res = self.compute_all_losses(train_batch)
        loss = res["loss"] + res["aux_loss"]
        pred = res["pred"]
        label = (2 - train_batch["is_correct"]) if not FIX_SUBTWO else train_batch["is_correct"]

        mask = train_batch["pad_mask"]
        infer_mask = train_batch["infer_mask"]

        nonzeros = torch.nonzero(infer_mask, as_tuple=True)
        pred = pred[nonzeros].sigmoid()
        label = label[nonzeros].long()
        if self.train_nonoverlap:
            loss = torch.sum(loss * mask * infer_mask) / torch.sum(mask * infer_mask)
        else:
            loss = torch.sum(loss * mask) / torch.sum(mask)
        self.train_preds.append(pred.detach().cpu())
        self.train_labels.append(label.detach().cpu())
        self.train_losses.append(loss.detach().cpu())

        return loss

    # def trainig_step_end(self, sub_batch_outputs):
    #     loss = torch.mean(sub_batch_outputs["loss"])
    #     if self.global_step % 1000 == 0:
    #         print(f"Epoch Train loss: {loss}")
    #         self.logger.experiment.log(
    #             {"Train loss": loss.item()}, step=self.global_step + 1
    #         )
    #     return {"loss": loss}

    # def training_epoch_end(self, training_step_outputs):
    #     # summarize train epoch results
    #     losses = []
    #     for out in training_step_outputs:
    #         losses.append(out["loss"])
    #     epoch_loss = torch.mean(torch.stack(losses))
    #
    #     self.log("train_loss", epoch_loss)
    #
    # def eval_step(self, batch):
    #     # val_step or test_step
    #     infer_result = self.inference(batch["feature"])
    #     label = infer_result["label"]  # (B, L)
    #     logit = infer_result["logit"]  # (B, L)
    #     infer_mask = batch["feature"]["infer_mask"]
    #
    #     bce_loss = self.compute_loss(logit, label, infer_mask)
    #     loss = bce_loss
    #
    #     nonzeros = torch.nonzero(infer_mask, as_tuple=True)
    #     pred = logit[nonzeros].sigmoid()
    #     label = label[nonzeros].long()
    #     return {"loss": loss, "pred": pred, "label": label}
    #
    # def eval_step_end(self, sub_batch_outputs):
    #     # val_step_end or test_step_end
    #     loss = torch.mean(sub_batch_outputs["loss"])
    #     pred, label = sub_batch_outputs["pred"], sub_batch_outputs["label"]
    #     return {"loss": loss, "pred": pred, "label": label}
    #
    # def eval_epoch_end(self, step_outputs):
    #     # val_epoch_end or test_epoch_end
    #     # summarize epoch results
    #     losses = []
    #     preds = []
    #     labels = []
    #     for out in step_outputs:
    #         losses.append(out["loss"])
    #         preds.append(out["pred"])
    #         labels.append(out["label"])
    #
    #     pred = torch.cat(preds, dim=0).view(-1)
    #     label = torch.cat(labels, dim=0).view(-1)
    #     loss = torch.mean(torch.stack(losses))
    #     auc = self.compute_roc_auc(label, pred)
    #     return {"loss": loss, "auc": auc}

    # def validation_step(self, batch, batch_idx):
    #     res = self.eval_step(batch)
    #     self.log(
    #         "validation_loss", res["loss"], on_step=True, on_epoch=True, sync_dist=True
    #     )
    #     return res
    #
    # def validation_step_end(self, sub_batch_outputs):
    #     return self.eval_step_end(sub_batch_outputs)

    # def validation_epoch_end(self, step_outputs):
    #     res = self.eval_epoch_end(step_outputs)
    #     loss, auc = res["loss"], res["auc"]
    #     print(
    #         f"Val loss: {float(loss):.4f}, auc: {auc:.4f}, previous best auc: {self.best_val_auc:.4f}"
    #     )
    #     if auc > self.best_val_auc:
    #         # update max valid auc & epoch and save weight
    #         self.best_val_auc = auc
    #
    #     if self.use_wandb:
    #         self.logger.experiment.log(
    #             {"Val loss": loss, "Val auc": auc, "Best Val auc": self.best_val_auc,},
    #             step=self.global_step + 1,
    #         )
    #     self.log("val_auc", auc)
    #
    # def test_step(self, batch, batch_idx):
    #     res = self.eval_step(batch)
    #     self.log("test_loss", res["loss"], on_step=True, on_epoch=True, sync_dist=True)
    #     return res
    #
    # def test_step_end(self, sub_batch_outputs):
    #     return self.eval_step_end(sub_batch_outputs)
    #
    # def test_epoch_end(self, step_outputs):
    #     res = self.eval_epoch_end(step_outputs)
    #     loss, auc = res["loss"], res["auc"]
    #
    #     print(
    #         f"Test loss: {float(loss):.4f}, auc: {auc:.4f}, previous best auc: {self.best_val_auc:.4f}"
    #     )
    #     if self.use_wandb:
    #         self.logger.experiment.log(
    #             {"Test loss": loss, "Test auc": auc,}
    #         )

