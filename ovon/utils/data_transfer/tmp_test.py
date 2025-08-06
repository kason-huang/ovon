from transformers import AutoProcessor, AutoModel

model = AutoModel.from_pretrained("google/siglip-base-batch16-224")
