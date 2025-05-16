# Load model 
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

processor = AutoImageProcessor.from_pretrained("MadanKhatri/finetuned-occupations")
model = AutoModelForImageClassification.from_pretrained("MadanKhatri/finetuned-occupations")

# Load the image
#Provide path to your images
image = Image.open("test/test5.png").convert("RGB")  
encoding = processor(image, return_tensors="pt")  
# print(encoding.pixel_values.shape)

# Forward pass and prediction
with torch.no_grad():
    outputs = model(**encoding)
    logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])