import torch.nn as nn
import torch
import numpy as np
from open_clip import create_model_and_transforms, get_tokenizer
from habitat_baselines.common.tensor_dict import TensorDict
from torchvision import transforms
from PIL import Image

class GoatSigLIPEncoder(nn.Module):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        model_name="hf-hub:timm/ViT-B-16-SigLIP-256"
        self.model, self.preprocess_train, self.preprocess_val = create_model_and_transforms(
            model_name = model_name,
        )
        self.model = self.model.to(self.device)

        self.tokenizer = get_tokenizer(model_name)
        self.output_size = 768
        self.output_shape = (self.output_size, )
        # ToPIL 用于把张量/ndarray转为 PIL（open_clip 的 preprocess 期望 PIL）
        self.to_pil = transforms.ToPILImage()

    # forward for the rgb observeration 
    def forward(self, observations: TensorDict, *args, **kwargs) -> torch.Tensor:
        preprocess = self.preprocess_val
        # 其实一直是冻结的，所以都是false
        if self.model.training:
            preprocess = self.preprocess_train

        rgb = observations["rgb"]  # [B, H, W, 3]
        assert rgb.ndim == 4 and rgb.shape[-1] == 3, "rgb should be [B,H,W,3]"

        # 确保在 CPU 上做 PIL 转换（PIL 不支持 GPU Tensor）
        rgb_cpu = rgb.detach().to("cpu")

        batch_imgs = []
        B = rgb_cpu.shape[0]
        for i in range(B):
            x = rgb_cpu[i]  # [H, W, 3]
            if x.dtype == torch.uint8:
                # 直接转 PIL
                pil = self.to_pil(x.permute(2, 0, 1))  # ToPILImage 期望 [C,H,W]
            else:
                # 浮点 [0,1]，先裁剪到合法范围再转 PIL
                x = x.clamp(0.0, 1.0)
                pil = self.to_pil(x.permute(2, 0, 1))

            # 使用 open_clip 提供的验证时预处理（与其权重严格匹配）
            t = preprocess(pil)  # [3, 256, 256]
            batch_imgs.append(t)

        # 组 batch 并送入设备
        x = torch.stack(batch_imgs, dim=0).to(self.device, dtype=torch.float32)  # [B,3,256,256]

        # encode_image：返回投影后且 L2 归一化的 embedding
        features = self.model.encode_image(x)  # [B, 768] (已 norm)
        return features
        
    def encode_image(self, image_input):
        preprocess = self.preprocess_val
        if self.model.training:
            preprocess = self.preprocess_train

        if isinstance(image_input, np.ndarray):
            image_input = Image.fromarray(image_input.astype(np.uint8)) 
        
        image_input = preprocess(image_input) 
        

        if len(image_input.shape) != 4:
            image_input = image_input.unsqueeze(0)
        # image_input= image_input.permute(0, 3, 1, 2)  # NHWC -> NCHW, have been done in preprocess

        image_input = image_input.to(self.device)
        # we do not want to update the encoder weight
        with torch.inference_mode():
            # Embedding will be 1x768
            features = self.model.encode_image(image_input)
        
        # features = features / features.norm(dim=-1, keepdim=True)

        return features

    def encode_text(self, text):
        # we do not want to update the encoder weight
        with torch.inference_mode():
            text_tokens = self.tokenizer(text).to(self.device)
            features = self.model.encode_text(text_tokens)

        # features = features / features.norm(dim=-1, keepdim=True)
        return features
    

    def batch_encode_images(
        self,
        images,                # List[np.ndarray | PIL.Image | torch.Tensor]
        batch_size: int = 128,
        return_numpy: bool = True,
    ):
        if not images:
            return np.empty((0, 0), dtype=np.float32) if return_numpy else torch.empty(0, 0)

        preprocess = self.preprocess_val
        if self.model.training:
            preprocess = self.preprocess_train

        all_feats = []
        with torch.inference_mode():
            for s in range(0, len(images), batch_size):
                batch_imgs = images[s:s+batch_size]
                prepped = []
                for img in batch_imgs:
                    if isinstance(img, np.ndarray):
                        img = Image.fromarray(img.astype(np.uint8)) 
                    t = preprocess(img)          # -> CHW
                    prepped.append(t)
                x = torch.stack(prepped, dim=0).to(self.device, non_blocking=True)  # [B,C,H,W]
                feats = self.model.encode_image(x)                                   # [B,D]
                all_feats.append(feats.detach().cpu())
        feats_all = torch.cat(all_feats, dim=0)  # [N,D]
        return feats_all.numpy() if return_numpy else feats_all



