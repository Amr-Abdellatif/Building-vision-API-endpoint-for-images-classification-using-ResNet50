import uvicorn
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import numpy as np
import onnxruntime as ort
import torchvision.transforms as transforms
import warnings
import onnx
from pipeline import convert_and_download
import os

app = FastAPI()

def check_and_convert():
    """ this basically checks for the resnet50 in onnx format
    if not avaliable then it brings the resnet50 from pytorch
    model zoo and turns it into onnx format """
    
    if not os.path.isfile("resnet50.onnx"):
        print('model not found, downloading and converting')
        convert_and_download()
    else:
        print('model exists')


# Suppress all warnings
warnings.filterwarnings("ignore")

onnx_model = onnx.load("./resnet50.onnx")
ort_session = ort.InferenceSession("resnet50.onnx")


# Load class labels
imagenet_labels_path = "imagenet-classes.txt"
with open(imagenet_labels_path) as f:
    labels = [line.strip() for line in f.readlines()]

# Image preprocessing
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

# Classify an example image using the ONNX model
def classify_image(image_path, topk=5):
    input_data = preprocess_image(image_path)
    ort_inputs = {ort_session.get_inputs()[0].name: input_data.numpy()}
    outputs = ort_session.run(None, ort_inputs)[0]
    top_classes = np.argsort(outputs)[0, ::-1][:topk]
    top_labels = [labels[i] for i in top_classes]
    return top_labels

@app.get("/")
async def home():
    return {'Home': 'Hello, World!'}



@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    classes = classify_image(image)
    return {"predictions": classes}

if __name__ == "__main__":
    check_and_convert()
    uvicorn.run(app, host="127.0.0.1", port=8000)