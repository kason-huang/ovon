import torch
from PIL import Image
import timm
import open_clip

# ---------- 1) 加载模型 ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

# open_clip：用官方预训练 + 自带预处理
m1, _, preprocess1 = open_clip.create_model_and_transforms(
    "ViT-B-16-SigLIP-256", pretrained="webli"  # 或你需要的权重; 你也可以用 hf-hub 路径
)
m1 = m1.to(device=device, dtype=dtype).eval()

# timm：确保 num_classes=0（返回特征），并保留 forward_features / proj
m2 = timm.create_model("vit_base_patch16_siglip_256", pretrained=True, num_classes=0)
m2 = m2.to(device=device, dtype=dtype).eval()

# timm 的预处理（用模型默认配置生成，避免细节不一致）
from timm.data import resolve_data_config, create_transform
cfg = resolve_data_config({}, model=m2)
preprocess2 = create_transform(**cfg)
# 若你想强行与 open_clip 一致，也可直接用 preprocess1 给 m2，但此处用 timm 官方更贴合其实现

# ---------- 2) 读图、预处理 ----------
img = Image.open("./debug/tmp_111.png").convert("RGB")
x1 = preprocess1(img).unsqueeze(0).to(device=device, dtype=dtype)
x2 = preprocess2(img).unsqueeze(0).to(device=device, dtype=dtype)

# ---------- 3) 取同语义层向量 ----------
with torch.no_grad():
    # open_clip：通常已投影并做过 L2
    z1 = m1.encode_image(x1)

    # timm：标准两步 -> features -> head（内部会做pooling和proj）
    feats = m2.forward_features(x2)                 # 不要自己调 global_pool
    z2 = m2.forward_head(feats, pre_logits=False)   # False 表示包含最终投影/头部

    # 显式 L2 归一化对齐 open_clip
    z1 = torch.nn.functional.normalize(z1, dim=-1)
    z2 = torch.nn.functional.normalize(z2, dim=-1)

# ---------- 4) 相似度 ----------
cos = (z1 * z2).sum(dim=-1).item()
print("cosine similarity:", cos)
print("shapes:", z1.shape, z2.shape)