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
        self.preds = []
        self.labels = []

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

        obs_mask = batch_dict["pad_mask"].to(device)  # padding mask
        src = self.embed["qid"](batch_dict["qid"].to(device))
        src += self.embed["skill"](batch_dict["skill"].squeeze(-1).to(device))
        # src += self.embed['part'](batch_dict['part'])
        # positional encoding
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
        
        # return {"loss": loss.mean()}
        return loss

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


class SAKT(SAINT, pl.LightningModule):
    def __init__(self, config):
        # super().__init__()
        pl.LightningModule.__init__(self)
        self.config = config

        self.embed = nn.ModuleDict()
        # maybe TODO: manage them with feature classes
        self.embed["qid"] = nn.Embedding(
            config.num_item + 1, config.dim_model, padding_idx=0
        )
        self.lin_in = nn.Linear(config.dim_model * 2, config.dim_model)
        if hasattr(config, "num_part"):
            self.embed["part"] = nn.Embedding(
                config.num_part + 1, config.dim_model, padding_idx=0
            )
        self.embed["skill"] = nn.Embedding(
            config.num_skill + 1, config.dim_model, padding_idx=0
        )
        self.embed["is_correct"] = nn.Embedding(3, config.dim_model, padding_idx=0)

        # transformer
        self.encoder_layers = clone(NonSelfAttentionLayer(
            d_model=config.dim_model,
            nhead=config.head_count,
            dim_feedforward=config.dim_ff,
            dropout=config.dropout_rate,
        ), config.layer_count * 2)
        self.layer_norms = clone(nn.LayerNorm(config.dim_model), config.layer_count * 2)
        # positional encoding
        self.embed["enc_pos"] = AbsoluteDiscretePositionalEncoding(
            dim_emb=config.dim_model, max_len=config.seq_len, device=config.device
        )
        self.generator = nn.Linear(config.dim_model, 1)

        self.val_auc = 0
        self.best_val_auc = 0
        self.best_step = -1
        self.test_auc = 0
        self.preds = []
        self.labels = []

        # xavier initialization
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
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
        keyval[..., : self.config.dim_model] *= torch.relu(shifted_infos["is_correct"] - 1).unsqueeze(-1)
        keyval[..., self.config.dim_model:] *= (1 - torch.relu(shifted_infos["is_correct"] - 1)).unsqueeze(-1)
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


class SAKT_(nn.Module):
    def __init__(
        self,
        num_items,
        num_skills,
        embed_size,
        num_attn_layers,
        num_heads,
        encode_pos,
        max_pos,
        drop_prob,
        query_feed=False,
        query_highpass=False,
        dim_ffw=128,
        max_seq_len=100
    ):
        """Self-attentive knowledge tracing.

        Arguments:
            num_items (int): number of items
            num_skills (int): number of skills
            embed_size (int): input embedding and attention dot-product dimension
            num_attn_layers (int): number of attention layers
            num_heads (int): number of parallel attention heads
            encode_pos (bool): if True, use relative position embeddings
            max_pos (int): number of position embeddings to use
            drop_prob (float): dropout probability
        """
        super(SAKT, self).__init__()
        self.embed_size = embed_size
        self.encode_pos = encode_pos
        self.query_feed = query_feed
        self.query_highpass = query_highpass
        self.max_seq_len = max_seq_len

        if 1:
            self.item_embeds = nn.Embedding(
                num_items + 1, embed_size // 2, padding_idx=0
            )
            self.skill_embeds = nn.Embedding(
                num_skills + 1, embed_size // 2, padding_idx=0
            )
        else:
            self.item_embeds = nn.Embedding(num_items + 1, embed_size, padding_idx=0)
            self.skill_embeds = nn.Embedding(num_skills + 1, embed_size, padding_idx=0)

        self.embed_pos = AbsoluteDiscretePositionalEncoding(
            dim_emb=embed_size, max_len=max_seq_len, device='cuda'
        )

        self.pos_key_embeds = nn.Embedding(max_pos, embed_size // num_heads)
        self.pos_value_embeds = nn.Embedding(max_pos, embed_size // num_heads)

        if 0:
            self.lin_in = nn.Linear(2 * embed_size, embed_size)
            self.attn_layers = clone(
                MultiHeadedAttention(embed_size, num_heads, drop_prob), num_attn_layers
            )
            self.layer_norms = clone(nn.LayerNorm(embed_size), num_attn_layers)
            self.dropouts = clone(nn.Dropout(p=drop_prob), num_attn_layers)
            self.lin_out = nn.Linear(embed_size, 1)
            # self.lin_out = nn.Linear(embed_size, num_skills)
        self.lin_in = nn.Linear(2 * embed_size, embed_size)
        self.encoder_layer = NonSelfAttentionLayer(embed_size, num_heads, dim_ffw, \
            dropout=drop_prob)
        self.lin_out = nn.Linear(embed_size, 1)

    def get_inputs(self, item_inputs, skill_inputs, label_inputs):
        item_inputs = self.item_embeds(item_inputs)
        skill_inputs = self.skill_embeds(skill_inputs)
        label_inputs = label_inputs.unsqueeze(-1).float()

        inputs = torch.cat(
            [item_inputs, skill_inputs, item_inputs, skill_inputs], dim=-1
        )

        inputs[..., : self.embed_size] *= label_inputs
        inputs[..., self.embed_size :] *= 1 - label_inputs
        return inputs  # Interaction: For Key and Value

    def get_query(self, item_ids, skill_ids):
        if 1:
            item_ids = self.item_embeds(item_ids)
            skill_ids = self.skill_embeds(skill_ids)
            query = torch.cat([item_ids, skill_ids], dim=-1)
            return query
        else:
            skill_ids = self.skill_embeds(skill_ids)
            return skill_ids  # Exercise: For Query

    def forward(self, item_inputs, skill_inputs, label_inputs, item_ids, skill_ids):
        inputs = self.get_inputs(item_inputs, skill_inputs, label_inputs)
        inputs = self.lin_in(inputs)
        # inputs = F.relu(self.lin_in(inputs))

        query = self.get_query(item_ids, skill_ids)

        mask = future_mask(inputs.size(-2)).squeeze(0)
        if inputs.is_cuda:
            mask = mask.cuda()
        
        if 0:
            attn_output = self.attn_layers[0](
                query, inputs, inputs, self.encode_pos, 
                self.pos_key_embeds, self.pos_value_embeds, mask,
                )
            if self.query_feed:
                attn_output = attn_output + query
            attn_output = self.layer_norms[0](attn_output)
            outputs = self.dropouts[0](attn_output)

            for i, l in enumerate(self.attn_layers[1:]):
                residual = l(
                    outputs if not self.query_highpass else query,
                    outputs,
                    outputs,
                    self.encode_pos,
                    self.pos_key_embeds,
                    self.pos_value_embeds,
                    mask,
                )
                outputs = self.dropouts[i + 1](self.layer_norms[i+1](outputs + F.relu(residual)))
        else:
            timestamp = torch.arange(1, self.max_seq_len + 1).to('cuda')
            enc_pos = self.embed_pos(timestamp).squeeze(0)  # (L_T, D)
            inputs += enc_pos
            query += enc_pos
            encoder_output = self.encoder_layer((query, inputs, inputs), mask)

        return self.lin_out(encoder_output)
