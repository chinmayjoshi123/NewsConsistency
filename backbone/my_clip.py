
import clip
import torch.nn as nn
import torch

class FeedForwardNeural(nn.Module):
    def __init__(self, input_size , hidden_layer, output_size , num_mlp = 4) -> None:
        super(FeedForwardNeural, self).__init__()
        mlps = []
        self.input_size = input_size
        self.hidden_layer = hidden_layer
        self.output_size  = output_size
        for _ in range(num_mlp - 1):
            mlps.append(nn.Linear(in_features= input_size , out_features= hidden_layer))
            mlps.append(nn.ReLU())
            input_size = hidden_layer
        mlps.append(nn.Linear(in_features= hidden_layer , out_features= output_size))
        self.mlp = nn.Sequential(*mlps)
    
    def forward(self , x):
        x = self.mlp(x)
        return x

class CLIP(nn.Module):
    def __init__(self) -> None:
        super(CLIP, self).__init__()
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        self.model ,self.preprocess  = clip.load('RN50' , device = self.device)
        self.model.visual.attnpool= nn.Identity()
        self.feed_forward = FeedForwardNeural(1024,4096,49)
        self.out_channels = 2048

    def forward(self ,image , caption):
        image_features = self.model.encode_image(image).to(self.device)
        text_features = clip.tokenize(caption,truncate = True).to(self.device)
        text_features = self.model.encode_text(text_features)
        text_features = self.feed_forward(text_features)
        text_features = text_features.reshape(text_features.shape[0],7,7).unsqueeze(1)
        combined_features = text_features + image_features
        return combined_features

def clip_encoder():
    return CLIP()

            




    