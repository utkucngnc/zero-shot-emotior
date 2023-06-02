from config import model, processor
from PIL import Image
import requests
import torch


url = "https://www.allprodad.com/wp-content/uploads/2021/03/05-12-21-happy-people.jpg"
image = Image.open(requests.get(url, stream=True).raw)

candidate_labels = ["happy", "sad", "angry", "shocked", "neutral"]

inputs = processor(images=image, text=candidate_labels, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits_per_image[0]
probs = logits.softmax(dim=-1).numpy()
scores = probs.tolist()

result = [
    {"score": score, "label": candidate_label}
    for score, candidate_label in sorted(zip(probs, candidate_labels), key=lambda x: -x[0])
]