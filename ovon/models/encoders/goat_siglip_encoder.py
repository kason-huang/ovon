import torch.nn as nn
import torch
from open_clip import create_model_from_pretrained, get_tokenizer

class GoatSigLIPEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        model_name="vit_base_patch16_siglip_256",
        self.model, self.preprocess_train, self.preprocess_val = create_model_from_pretrained(
            model_name = model_name,
            pretrained=True,
            num_classes=0,
        )

        self.tokenizer = get_tokenizer(model_name)
    
    def encode_image(self, image_input):
        preprocess = self.preprocess_val
        if self.model.training:
            preprocess = self.preprocess_train
        
        # we do not want to update the encoder weight
        image_input= image_input.permute(0, 3, 1, 2)  # NHWC -> NCHW

        with torch.inference_mode():
            x = preprocess(image_input)
            features = self.model.encode_image(x)
        
        features /= features.norm(dim=-1, keepdim=True)

        return features

    def encode_text(self, text):
        # we do not want to update the encoder weight
        with torch.inference_mode():
            text_tokens = self.tokenizer(text)
            feature = self.model.encode_text(text_tokens)

        features /= features.norm(dim=-1, keepdim=True)
        return features



