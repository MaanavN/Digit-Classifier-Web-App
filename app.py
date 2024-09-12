from flask import Flask, render_template, request, redirect, url_for
from PIL import Image, ImageOps
import io
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model import classify_image

app = Flask(__name__)

# Define transformation: resize to 28x28 and convert to grayscale
transform = transforms.Compose([
    transforms.ToTensor()
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Open the image
        image = Image.open(io.BytesIO(file.read()))
        image = image.resize((28, 28))
        image = image.rotate(-90)
        image = image.convert("L")
        image = ImageOps.invert(image)
        image.show()
        image = image.point(lambda p: 255 if p >= 127.5 else 0)
        image.show()

        # Apply the transformations
        image_tensor = transform(image)
        image_tensor = image_tensor.view(-1, 784)

        # Log the tensor shape for debugging purposes
        print(f"Image Tensor Shape: {image_tensor.shape}")


        output = classify_image(image_tensor)
        print(f"Prediction: {output}")
        
        return f"Prediction: {output}"

if __name__ == '__main__':
    app.run(debug=True)
