import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.transformersv2 import (
    AtomFeature,
    EdgeFeature,
    SE3InvariantKernel,
    MovementPredictionHead,
    TransformerEncoderWithPairV2,
)
from ..utils import pad_1d_tokens, pad_coords, pad_2d


class UniMolV2Model(nn.Module):
    """Pretraining model for UniMol2."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_num = 128
        self.padding_idx = 0
        self.mask_idx = self.token_num - 1

        self.embed_tokens = nn.Embedding(
            self.token_num, config.encoder_embed_dim, self.padding_idx
        )

        self.atom_feature = AtomFeature(
            num_atom=512,
            num_degree=128,
            hidden_dim=config.encoder_embed_dim,
        )

        self.edge_feature = EdgeFeature(
            pair_dim=config.pair_embed_dim,
            num_edge=64,
            num_spatial=512,
        )

        self.encoder = TransformerEncoderWithPairV2(
            num_encoder_layers=config.encoder_layers,
            embedding_dim=config.encoder_embed_dim,
            pair_dim=config.pair_embed_dim,
            pair_hidden_dim=config.pair_hidden_dim,
            ffn_embedding_dim=config.encoder_ffn_embed_dim,
            num_attention_heads=config.encoder_attention_heads,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation_dropout=config.activation_dropout,
            activation_fn=config.activation_fn,
            droppath_prob=getattr(config, "droppath_prob", 0.0),
            pair_dropout=getattr(config, "pair_dropout", 0.0),
        )

        K = 128
        self.se3_invariant_kernel = SE3InvariantKernel(
            pair_dim=config.pair_embed_dim,
            num_pair=512,
            num_kernel=K,
            std_width=getattr(config, "gaussian_std_width", 1.0),
            start=getattr(config, "gaussian_mean_start", 0.0),
            stop=getattr(config, "gaussian_mean_stop", 9.0),
        )

        self.movement_pred_head = MovementPredictionHead(
            config.encoder_embed_dim,
            config.pair_embed_dim,
            config.encoder_attention_heads,
        )

        if config.masked_token_loss > 0:
            self.lm_head = MaskLMHead(
                embed_dim=config.encoder_embed_dim,
                output_dim=self.token_num,
                activation_fn=config.activation_fn,
            )

        self.classification_heads = nn.ModuleDict()
        self.dtype = torch.float32

    def forward(
        self,
        src_tokens,
        src_coord,
        atom_feat,
        atom_mask,
        pair_type,
        attn_bias,
        edge_feat,
        shortest_path,
        degree,
        encoder_masked_tokens=None,
        **kwargs,
    ):
        token_feat = self.embed_tokens(src_tokens)
        batched_data = {
            "atom_feat": atom_feat,
            "degree": degree,
            "edge_feat": edge_feat,
            "shortest_path": shortest_path,
            "attn_bias": attn_bias,
        }

        x = self.atom_feature(batched_data, token_feat)

        attn_mask = attn_bias.clone()
        attn_bias_t = torch.zeros_like(attn_mask)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.config.encoder_attention_heads, 1, 1)
        attn_bias_t = attn_bias_t.unsqueeze(-1).repeat(1, 1, 1, self.config.pair_embed_dim)
        attn_bias_t = self.edge_feature(batched_data, attn_bias_t)
        attn_mask = attn_mask.type(self.dtype)

        n_mol = src_tokens.size(0)
        atom_mask_cls = torch.cat(
            [torch.ones(n_mol, 1, device=atom_mask.device, dtype=atom_mask.dtype), atom_mask],
            dim=1,
        ).type(self.dtype)
        pair_mask = atom_mask_cls.unsqueeze(-1) * atom_mask_cls.unsqueeze(-2)

        pos = src_coord

        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = delta_pos.norm(dim=-1)
        attn_bias_3d = self.se3_invariant_kernel(dist.detach(), pair_type)
        new_attn_bias = attn_bias_t.clone()
        new_attn_bias[:, 1:, 1:, :] = new_attn_bias[:, 1:, 1:, :] + attn_bias_3d
        new_attn_bias = new_attn_bias.type(self.dtype)

        x, pair = self.encoder(
            x,
            new_attn_bias,
            atom_mask=atom_mask_cls,
            pair_mask=pair_mask,
            attn_mask=attn_mask,
        )

        node_output = self.movement_pred_head(
            x[:, 1:, :],
            pair[:, 1:, 1:, :],
            attn_mask[:, :, 1:, 1:],
            delta_pos.detach(),
        )

        pos = pos + node_output

        logits = None
        encoder_coord = None
        encoder_distance = None

        if self.config.masked_token_loss > 0:
            logits = self.lm_head(x[:, 1:, :], encoder_masked_tokens)
        if self.config.masked_coord_loss > 0:
            encoder_coord = pos
        if self.config.masked_dist_loss > 0:
            encoder_distance = (pos.unsqueeze(1) - pos.unsqueeze(2)).norm(dim=-1)

        return logits, encoder_distance, encoder_coord, None, None

    def batch_collate_fn(self, batch):
        net_input = {
            'src_tokens': pad_1d_tokens([item[0]['src_tokens'] for item in batch], self.padding_idx),
            'src_coord': pad_coords([item[0]['src_coord'] for item in batch], pad_idx=0.0),
            'src_distance': pad_2d([item[0]['src_distance'] for item in batch], pad_idx=0.0),
            'src_edge_type': pad_2d([item[0]['src_edge_type'] for item in batch], self.padding_idx),
            'atom_feat': pad_1d_feat([item[0]['atom_feat'] for item in batch]),
            'atom_mask': pad_1d_tokens([item[0]['atom_mask'] for item in batch], 0),
            'edge_feat': pad_2d_feat([item[0]['edge_feat'] for item in batch]),
            'shortest_path': pad_2d([item[0]['shortest_path'] for item in batch], pad_idx=0),
            'degree': pad_1d_tokens([item[0]['degree'] for item in batch], 0),
            'pair_type': pad_2d_feat([item[0]['pair_type'] for item in batch], pad_idx=0),
            'attn_bias': pad_attn_bias([item[0]['attn_bias'] for item in batch]),
        }
        net_target = {
            'tgt_tokens': pad_1d_tokens([item[1]['tgt_tokens'] for item in batch], self.padding_idx),
            'tgt_coordinates': pad_coords([item[1]['tgt_coordinates'] for item in batch], pad_idx=0.0),
            'tgt_distance': pad_2d([item[1]['tgt_distance'] for item in batch], pad_idx=0.0),
        }
        return {'net_input': net_input, 'net_target': net_target}


class MaskLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = getattr(F, activation_fn) if isinstance(activation_fn, str) else activation_fn
        self.layer_norm = nn.LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = F.linear(x, self.weight) + self.bias
        return x


def pad_1d_feat(samples, pad_idx=0):
    max_len = max(s.size(0) for s in samples)
    feat_dim = samples[0].size(-1)
    out = samples[0].new_full((len(samples), max_len, feat_dim), pad_idx)
    for i, s in enumerate(samples):
        out[i, : s.size(0)] = s
    return out


def pad_2d_feat(samples, pad_idx=0):
    max_len = max(s.size(0) for s in samples)
    feat_dim = samples[0].size(-1)
    out = samples[0].new_full((len(samples), max_len, max_len, feat_dim), pad_idx)
    for i, s in enumerate(samples):
        out[i, : s.size(0), : s.size(1)] = s
    return out


def pad_attn_bias(samples):
    max_len = max(s.size(0) - 1 for s in samples)
    out = samples[0].new_full((len(samples), max_len + 1, max_len + 1), float("-inf"))
    for i, s in enumerate(samples):
        out[i, : s.size(0), : s.size(1)] = s
        out[i, s.size(0):, : s.size(1)] = 0
    return out
