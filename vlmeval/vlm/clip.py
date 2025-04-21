import torch
from PIL import Image
import open_clip
import clip

class CLIPVLMWrapper(torch.nn.Module):
    def __init__(self, model_path='openai/clip-vit-base-patch32', **kwargs):
        super(CLIPVLMWrapper, self).__init__()
        self.model_path = model_path
        self.eval_type = kwargs['eval_type']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.preprocess = CLIPProcessor.from_pretrained(model_path)
    
    def __call__(self, captions, images):
        images = Image.open(images).convert("RGB")
        inputs = self.preprocess(text=captions, images=images, return_tensors="pt", padding=True, do_rescale=False)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits_per_image  # this is the image-text similarity score
        if self.eval_type == 'singlelabel':
            probs = logits.softmax(dim=1)
        elif self.eval_type == 'sigmoid':
            probs = (logits/100).sigmoid()
        return probs.cpu().numpy()
    


class CLIPOpenAIWrapper(torch.nn.Module):
    def __init__(self, model_path='ViT-B/32', **kwargs):
        super(CLIPOpenAIWrapper, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.eval_type = kwargs['eval_type']
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
        self.model_path = model_path
        self.model.to(self.device)
    
    def __call__(self, captions, images):
        images = Image.open(images).convert("RGB")
        images = self.preprocess(images).unsqueeze(0).to(self.device)
        text = clip.tokenize(captions).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T)
            if self.eval_type == 'singlelabel':
                probs = similarity.softmax(dim=-1)
            elif self.eval_type == 'sigmoid':
                probs = (similarity/100).sigmoid()
            elif self.eval_type == 'negative_examples':
                probs = similarity
            return probs.cpu().numpy()
