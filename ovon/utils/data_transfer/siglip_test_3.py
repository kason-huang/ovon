# code https://huggingface.co/docs/transformers/en/model_doc/siglip?usage=AutoModel
# 这是有版本要求的，哈哈
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModel

model = AutoModel.from_pretrained("google/siglip-base-patch16-224", torch_dtype=torch.float16, device_map="auto", attn_implementation="sdpa")
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

#image_path = "./debug/tmp_111.png"  # 替换为你的图片路径
#image = Image.open(image_path).convert("RGB")

candidate_labels = ["a Pallas cat", "a bathtub", "a Siberian tiger"]
texts = [f'This is a photo of {label}.' for label in candidate_labels]
inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**inputs)

# 获取图文 embedding
image_embeds = outputs.image_embeds
text_embeds = outputs.text_embeds

# 计算余弦相似度
similarity = torch.nn.functional.cosine_similarity(image_embeds, text_embeds)
print("Cosine similarity:", similarity)

# logits_per_image = outputs.logits_per_image
# probs = torch.sigmoid(logits_per_image)
# print(f"{probs[0][0]:.1%} that image 0 is '{candidate_labels[0]}'")