 # added by Anita Rau April 2025, based on the original SurgVLP repo 

try:
    import surgvlp
except ImportError:
    pass # surgvlp and VLMEvalKit have version conflicts
import torch

import torchvision.transforms as transforms
from mmengine.config import Config
from PIL import Image
"""
Project: Learning Multi-modal Representations by Watching Hundreds of Surgical Video Lectures
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
"""
config = dict(
    dataset_config=[
    dict(
    type='Recognition_frame',
    csv_root='./csvs',
    vid='video%02d.csv'%i,
    video_root='./tmp/tmp',
    transforms=transforms.Compose(
        [
        transforms.Resize((360, 640)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        ),
    ) for i in range(49, 81)
    ],
    model_config = dict(
        type='SurgVLP',
        backbone_img = dict(
            type='img_backbones/ImageEncoder',
            num_classes=768,
            pretrained='imagenet',
            backbone_name='resnet_50',
            img_norm=False
        ),
        backbone_text= dict(
            type='text_backbones/BertEncoder',
            text_bert_type='emilyalsentzer/Bio_ClinicalBERT',
            text_last_n_layers=4,
            text_aggregate_method='sum',
            text_norm=False,
            text_embedding_dim=768,
            text_freeze_bert=False,
            text_agg_tokens=True
        )
    )
)



class SurgVLPWrapper(torch.nn.Module):
    def __init__(self, **kwargs):
        super(SurgVLPWrapper, self).__init__()
        configs = Config(config)
        self.eval_type = kwargs['eval_type']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        import surgvlp
        self.model, self.preprocess = surgvlp.load(configs.model_config, device=self.device)
        self.model_path = configs.model_config.type
    
    def __call__(self, captions, images):
        images = Image.open(images).convert("RGB")
        image = self.preprocess(images).unsqueeze(0).to(self.device)
        import surgvlp
        text = surgvlp.tokenize(captions, device=self.device)      

        with torch.no_grad():
            output_dict = self.model(image, text , mode='all')

            image_embeddings = output_dict['img_emb']
            text_embeddings= output_dict['text_emb']

            image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

            logits_per_image = 100.0 * image_embeddings @ text_embeddings.T
            if self.eval_type == 'singlelabel':
                probs = logits_per_image.softmax(dim=-1)
            elif self.eval_type == 'sigmoid':
                probs = logits_per_image.sigmoid()
            return probs.cpu().numpy()
