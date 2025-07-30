import torch
from transformers import AutoProcessor, AutoModel
import pickle
import numpy as np

def extract_and_save_text_features(item_classes, save_path="siglip_6.pkl"):
    # 加载模型与处理器
    model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    
    # 提取文本特征（批量处理）
    text_inputs = processor(text=item_classes, padding=True, return_tensors="pt")
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs).cpu().numpy()
    
    # 构建Key-Value字典：{类别: 特征向量}
    feature_dict = {cls: feat for cls, feat in zip(item_classes, text_features)}
    
    # 保存为.pkl文件
    with open(save_path, "wb") as f:
        pickle.dump(feature_dict, f)
    print(f"特征已保存至 {save_path}，格式: {{类别: 向量}}")

# 示例调用
item_classes = ["chair", "bed", "plant", "toilet", "tv_monitor", "sofa"]  
extract_and_save_text_features(item_classes)