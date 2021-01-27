import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from vltk.modeling._vision_transformer import VisionTransformer


class DistilledVisionTransformer(VisionTransformer):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=784,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.10,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        hybrid_backbone=None,
        *args,
        **kwargs,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            hybrid_backbone=hybrid_backbone,
        )
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = (
            nn.Linear(self.embed_dim, self.num_classes)
            if self.num_classes > 0
            else nn.Identity()
        )

        trunc_normal_(self.dist_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        attns = None
        for blk in self.blocks:
            x, attn = blk(x)
            if attns is None:
                attns = attn.unsqueeze(1)
            else:
                attns = torch.cat((attns, attn.unsqueeze(1)), dim=1)

        x = self.norm(x)
        return {"cls": x[:, 0], "dist": x[:, 1], "feats": x[:, 2:], "attns": attns}

    def forward(self, x):
        x = self.forward_features(x)
        return x

        # x = self.head(x)
        # x_dist = self.head_dist(x_dist)
        # if self.training:
        #     return x, x_dist
        # else:
        #     # during inference, return the average of both classifier predictions
        #     return (x + x_dist) / 2
