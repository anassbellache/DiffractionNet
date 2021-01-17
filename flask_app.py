from flask import Flask, request, jsonify, render_template, redirect, url_for
from PIL import Image
import numpy as np
from torchvision import models, transforms
import re
import base64
import torch
from model.deep_freak import get_classifier
import io




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = get_classifier(num_classes=2, device=device)
model.load_state_dict(torch.load('./model/weights.pt'))
model.eval()

app = Flask(__name__)


class_index = {'0':'Weak diffraction signal', '1':'Strong diffraction signal'}


def stringToImage(img):
    imgstr = re.search(r'base64, (.*)', str(img)).group(1)
    with open('image.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))

def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0).to(device)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return predicted_idx, class_index[predicted_idx]


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        file_ = request.files['image_file']
        img_bytes = file_.read()
        _, class_name = get_prediction(img_bytes)
        return render_template('index.html', sig= class_name)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
    
