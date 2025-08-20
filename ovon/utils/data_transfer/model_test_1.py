# 1. 用 open_clip

from open_clip import create_model_from_pretrained
from PIL import Image
import torch

model1, _ = create_model_from_pretrained("hf-hub:timm/ViT-B-16-SigLIP-256")
model1.eval()

# 2. 用 timm
import timm
from torchvision import transforms

model2 = timm.create_model("vit_base_patch16_siglip_256", pretrained=True, num_classes=0)
model2.eval()

preprocess2 = transforms.Compose([
    transforms.Resize(
        size=(256, 256), interpolation=transforms.InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=torch.tensor([0.5000, 0.5000, 0.5000]),
        std=torch.tensor([0.5000, 0.5000, 0.5000]),
    )
])
preprocess1 = preprocess2

# 输入图像
image = Image.open("./debug/tmp_111.png").convert("RGB")

# 预处理
img1 = preprocess1(image).unsqueeze(0)
img2 = preprocess2(image).unsqueeze(0)

# 比较特征
with torch.no_grad():
    emb1 = model1.encode_image(img1)
    emb2 = model2(img2)

cosine_sim = torch.nn.functional.cosine_similarity(emb1, emb2)
print("Cosine similarity:", cosine_sim.item())
print(emb1.shape, emb2.shape)
print(torch.norm(emb1), torch.norm(emb2))
print(emb1[0, :5])
print(emb2[0, :5])
