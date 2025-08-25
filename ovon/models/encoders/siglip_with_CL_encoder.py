import timm
import torch
import torch.nn as nn
from habitat_baselines.common.tensor_dict import TensorDict
from torchvision import transforms
import math

class PatchReshape(nn.Module):
    """(N, L, D) -> (N, D, H, W), 其中 L = H*W 且假设 H=W"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, L, D = x.shape
        H = W = int(math.sqrt(L) + 0.5)
        assert H * W == L, f"L={L} 不是完全平方数，无法reshape成网格。"
        x = x.view(N, H, W, D).permute(0, 3, 1, 2).contiguous()  # (N,D,H,W)
        return x

class SigLIPWithCompressionLayerEncoder(nn.Module):
    def __init__(self, approx_output_size: int = 2048, fc_out_dim: int = 768):
        super().__init__()
        self.model = timm.create_model(
            model_name="vit_base_patch16_siglip_256",
            pretrained=True,
            num_classes=0,
        )
        self.model = self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.transforms = transforms.Compose(
            [
                transforms.Resize(
                    size=(256, 256), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.Normalize(
                    mean=torch.tensor([0.5000, 0.5000, 0.5000]),
                    std=torch.tensor([0.5000, 0.5000, 0.5000]),
                ),
            ]
        )
        # -------- compression layer 规格 --------------
        self.approx_output_size = approx_output_size
        self.embed_dim = getattr(self.model, "embed_dim", getattr(self.model, "num_features", 768))


        # 由输入尺寸与 patch 大小推 L（H*W），从而确定 out_channels
        patch = getattr(self.model.patch_embed, "patch_size", 16)
        if isinstance(patch, tuple):
            ph, pw = patch
        else:
            ph = pw = int(patch)
        gh = 256 // ph
        gw = 256 // pw
        self.num_patches = gh * gw                                 # L
        self.num_channels = int(round(self.approx_output_size / self.num_patches))

        # --- compression layer 本体 ---
        self.compression = nn.Sequential(
            PatchReshape(),                                         # (N,L,D)->(N,D,H,W)
            nn.Conv2d(self.embed_dim, self.num_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=self.num_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),                                           # (N, C', H, W)->(N, C'*H*W)
        )

        # CL 的真实输出维度（约等于 approx_output_size）
        self.cl_output_size = self.num_channels * self.num_patches

        # --- FC 头：把 CL 输出投到 768 ---
        self.head = nn.Sequential(
            nn.Linear(self.cl_output_size, fc_out_dim),
            nn.GELU(),
            nn.LayerNorm(fc_out_dim),
        )

        # 对外声明的输出形状
        self.output_size = fc_out_dim
        self.output_shape = (self.output_size,)
    
     # 取 ViT 的 patch token（不做全局池化），保持冻结
    def _forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        m = self.model
        x = m.patch_embed(x)                           # (B, L, D)
        # 位置编码（根据形状自动对齐；SigLIP通常无CLS或用avg pool）
        pe = getattr(m, "pos_embed", None)
        if pe is not None:
            if pe.shape[1] == x.shape[1]:
                x = x + pe
            elif pe.shape[1] == x.shape[1] + 1:       # 去掉可能的CLS位
                x = x + pe[:, 1:, :]
        if getattr(m, "pos_drop", None) is not None:
            x = m.pos_drop(x)
        for blk in m.blocks:
            x = blk(x)                                # 内含 MSA
        if getattr(m, "norm", None) is not None:
            x = m.norm(x)
        return x            

    def forward(self, observations: TensorDict, *args, **kwargs) -> torch.Tensor:
        rgb = observations["rgb"]
        rgb = rgb.permute(0, 3, 1, 2)  # NHWC -> NCHW
        if rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0
        else:
            assert (rgb >= 0.0).all() and (rgb <= 1.0).all()
        with torch.inference_mode():
            # rgb be of size Bx3x224x224
            x = self.transforms(rgb)
            # Embedding will be 1x768 （256 / 16 = 16, 所以是3X16X16 => 768,所以每个特种图其实压缩为1个元素了）
            tokens = self._forward_tokens(x)           # (N, L, D)

        # compression layer（可训练） + FC 投影到 768
        feat = self.compression(tokens)                # (N, ~approx_output_size)
        out = self.head(feat)                          # (N, 768)
        return out