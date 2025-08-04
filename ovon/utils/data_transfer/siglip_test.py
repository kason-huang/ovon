import torch
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer

# 1. 选择模型，推荐用一个比较通用的 open_clip 模型，比如 ViT-B-16
model_name = "hf-hub:timm/ViT-B-16-SigLIP-256"

# 2. 加载模型和tokenizer，preprocess函数
model, preprocess = create_model_from_pretrained(model_name)
tokenizer = get_tokenizer(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

# 3. 准备输入

# 读图片并预处理
image_path = "./debug/tmp_111.png"  # 替换为你的图片路径
image = Image.open(image_path).convert("RGB")
image_input = preprocess(image).unsqueeze(0)  # 增加 batch 维度


# 文本输入
texts = ["bathtub", "dog", "house", "white bathtub in bathroom", "a photo of a bathtub", "a photo of a whilte bathtub"]  # 你也可以加多条文本一起比较
text_tokens = tokenizer(texts)

image_input = image_input.to(device)
text_tokens = text_tokens.to(device)


# 4. 计算图片和文本的embedding
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_tokens)

# 5. 归一化向量
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

# 6. 计算余弦相似度（点积即可，因为已经归一化）
# similarity = (image_features @ text_features.T).item()

# 5. 相似度计算
similarities = (image_features @ text_features.T).squeeze(0)   # shape: [3]

# 6. 打印每个文本的相似度
for text, sim in zip(texts, similarities):
    print(f"Similarity to '{text}': {sim.item():.4f}")

# 7. 输出最相似类别
best_idx = similarities.argmax().item()
print(f"\nPrediction: {texts[best_idx]}")
