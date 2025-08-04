import torch
from PIL import Image
import open_clip

# 1. 加载 SigLIP 模型和预处理函数
model_name = "hf-hub:timm/ViT-B-16-SigLIP-256" # SigLIP ViT-B-16 模型示例
model, _, preprocess = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

# 2. 加载图片，做预处理
image_path = "./debug/tmp_111.png"
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

# 3. 文字数组
text_descriptions = [
    "bathtub",
    "dog",
    "a photo of a bathtub",
    "white bathtub",
]

# 4. 文本编码
text_tokens = tokenizer(text_descriptions).to(device)

with torch.no_grad():
    # 图片编码
    image_features = model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)

    # 文本编码
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # 计算余弦相似度，图片向量和每条文本向量
    similarity = (image_features @ text_features.T).squeeze(0)  # shape: (num_texts,)

# 5. 输出相似度
for desc, score in zip(text_descriptions, similarity.cpu().numpy()):
    print(f"描述: {desc}  --> 相似度: {score:.4f}")
