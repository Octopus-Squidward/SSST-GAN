import math
import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import drop_path, to_2tuple, trunc_normal_


def get_reference_points(spatial_shapes, device):
    H_, W_ = spatial_shapes[0], spatial_shapes[1]

    ref_y, ref_x = torch.meshgrid(
        torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
        torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))

    ref_y = ref_y.reshape(-1)[None] / H_
    ref_x = ref_x.reshape(-1)[None] / W_
    ref = torch.stack((ref_x, ref_y), -1)

    return ref


class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SampleAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, n_points=4):
        super().__init__()
        self.n_points = n_points
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)

        self.sampling_offsets = nn.Linear(all_head_dim, self.num_heads * n_points * 2)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W, deform_inputs):
        B, N, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, -1).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        reference_points, input_spatial_shapes = deform_inputs

        sampling_offsets = self.sampling_offsets(q).reshape(
            B, N, self.num_heads, self.n_points, 2).transpose(1, 2)

        _, _, L = q.shape

        q = q.reshape(B, N, self.num_heads, L // self.num_heads).transpose(1, 2)

        offset_normalizer = torch.stack([input_spatial_shapes[1], input_spatial_shapes[0]])

        sampling_locations = reference_points[:, None, :, None, :] \
                             + sampling_offsets / offset_normalizer[None, None, None, None, :]

        sampling_locations = 2 * sampling_locations - 1

        k = k.reshape(B, N, self.num_heads, L // self.num_heads).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, L // self.num_heads).transpose(1, 2)

        k = k.flatten(0, 1).transpose(1, 2).reshape(B * self.num_heads, L // self.num_heads, input_spatial_shapes[0],
                                                    input_spatial_shapes[1])
        v = v.flatten(0, 1).transpose(1, 2).reshape(B * self.num_heads, L // self.num_heads, input_spatial_shapes[0],
                                                    input_spatial_shapes[1])

        sampling_locations = sampling_locations.flatten(0, 1).reshape(B * self.num_heads, N, self.n_points, 2)

        q = q[:, :, :, None, :]

        sampled_k = F.grid_sample(k, sampling_locations, mode='bilinear',
                                  padding_mode='zeros', align_corners=False).reshape(B, self.num_heads,
                                                                                     L // self.num_heads, N,
                                                                                     self.n_points).permute(0, 1, 3, 4,
                                                                                                            2)

        sampled_v = F.grid_sample(v, sampling_locations, mode='bilinear',
                                  padding_mode='zeros', align_corners=False).reshape(B, self.num_heads,
                                                                                     L // self.num_heads, N,
                                                                                     self.n_points).permute(0, 1, 3, 4,
                                                                                                            2)

        attn = (q * sampled_k).sum(-1) * self.scale

        attn = attn.softmax(dim=-1)[:, :, :, :, None]

        x = (attn * sampled_v).sum(-2).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W, rel_pos_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B,H,N,N

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None, sample=False, n_points=None):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.sample = sample

        if not sample:
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)

        else:
            self.attn = SampleAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim,
                n_points=n_points)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, H, W, deform_inputs):
        if self.gamma_1 is None:
            if not self.sample:
                x = x + self.drop_path(self.attn(self.norm1(x), H, W))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x), H, W, deform_inputs))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if not self.sample:
                x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), H, W))
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), H, W, deform_inputs))
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        x = self.proj(x)

        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)

        return x, (Hp, Wp)


def deform_PatchInputs_func(x, seg_patch_size):
    B, c, h, w = x.shape

    spatial_shapes = torch.as_tensor([h // seg_patch_size, w // seg_patch_size],
                                     dtype=torch.long, device=x.device)

    reference_points = get_reference_points([h // seg_patch_size, w // seg_patch_size],
                                            x.device)

    deform_inputs = [reference_points, spatial_shapes]

    return deform_inputs


class SpatViT(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, use_checkpoint=False,
                 use_abs_pos_emb=False, use_rel_pos_bias=False,
                 out_indices=[11], interval=3, pretrained=None, restart_regression=True, n_points=4):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_features = self.embed_dim = embed_dim
        self.in_chans = in_chans

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.out_indices = out_indices

        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            self.pos_embed = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.use_rel_pos_bias = use_rel_pos_bias
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, sample=((i + 1) % interval != 0),
                restart_regression=restart_regression, n_points=n_points)
            for i in range(depth)])

        self.interval = interval

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        self.norm = norm_layer(embed_dim)

        self.apply(self._init_weights)
        self.fix_init_weight()
        self.pretrained = pretrained

        self.out_channels = (3, embed_dim, embed_dim, embed_dim, embed_dim)

    def fix_init_weight(self):

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        img = [x]

        deform_inputs = deform_PatchInputs_func(x, self.seg_patch_size)

        B, C, H, W = x.shape

        x, (Hp, Wp) = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        if self.pos_embed is not None:
            x = x + self.pos_embed

        x = self.pos_drop(x)

        features = []
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, Hp, Wp, deform_inputs)
            else:
                x = blk(x, Hp, Wp, deform_inputs)
            if i in self.out_indices:
                features.append(x)

        features = list(map(lambda x: x.permute(0, 2, 1).reshape(B, -1, Hp, Wp), features))

        return img + features

    def forward(self, x):
        x = self.forward_features(x)
        return x
