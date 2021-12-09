import streamlit as st
import random
import os
import time
import numpy as np
from PIL import Image
import torch
from model.deep_freak import get_classifier
from torchvision import transforms, utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.title("X-ray diffraction classification")

col1,col2 = st.beta_columns(2)
image = col1.file_uploader("Diffraction signal", type=['png','jpg','jpeg'])

col1,col2 = st.beta_columns(2)
test_data = col1.button('Predict on a random sample from dataset')

class_index = {'0':'Weak diffraction signal', '1':'Good diffraction signal'}

@st.cache
def load_model(model_path):
    with torch.no_grad():
        model = get_classifier(2, device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
    return model

def predict(image):
    if image is not None:
        image = Image.open(image)
        image = np.array(image)/255
        st.image(image, width=300)
        image = image.astype("float32")
        my_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor()])
        img_tensor = my_transforms(image).unsqueeze(0).to(device)
        model = load_model('./model/weights.pt')
        outputs = model.forward(img_tensor)
        _, y_hat = outputs.max(1)
        predicted_idx = str(y_hat.item())
        #print("predicted class: ", predicted_idx)
        st.markdown("##  {}".format(class_index[predicted_idx]))
    else:
        st.markdown("## Upload an Image")


def predict_sample(folder='./data/synthetic/test'):
    no_folders = len(os.listdir(folder))
    cl = np.random.randint(1,no_folders)
    folder_path = os.path.join(folder,str(cl))
    file_ = random.choice(os.listdir(folder_path))
    predict(os.path.join(folder_path, file_))


if test_data:
    predict_sample()
else:
    predict(image)

    
