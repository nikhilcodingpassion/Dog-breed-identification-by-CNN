from flask import Flask, render_template, request, redirect, url_for
import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import torch.nn as nn

import os

app = Flask(__name__)

# Load the trained model
def load_model(checkpoint_path):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 120)  

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)  # Load the model state dict

    model.eval()
    return model


model = load_model('./checkpoint.pth.tar')  
# Load class names
class_names = pd.read_csv('class_names.csv')['breed'].tolist() 

# Image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            image = Image.open(file.stream)
            image = transform(image).unsqueeze(0)
            if torch.cuda.is_available():
                image = image.cuda()

            with torch.no_grad():
                outputs = model(image)
                _, preds = torch.max(outputs, 1)
                predicted_breed = class_names[preds[0]]

            return render_template('index.html', predicted_breed=predicted_breed)
    return render_template('index.html', predicted_breed=None)

@app.route('/lost-and-found')
def lost_and_found():
    return render_template('lost_and_found.html')

# Load labels with breed information
labels_df = pd.read_csv('labels1.csv')
@app.route('/breed-education')
def breed_education():
    # Create a list of dictionaries with breed and image file path
    breeds_info = [
        {
            'breed': row['breed'],
            'image_file': row['id'] + '.jpg'  # Assuming images are in .jpg format
        } for index, row in labels_df.iterrows()
    ]
    return render_template('breed_education.html', breeds_info=breeds_info)




if __name__ == '__main__':
    app.run(debug=True)