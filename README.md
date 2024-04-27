# Main thoughts

Building API-endpoint for image classification, when running main.py it checks for the model in onnx format in your directory if it's not there already it will downlaod it from torch model zoo and change it into onnx format and then the API is in ACTION !!!

## Preview

https://github.com/Amr-Abdellatif/Building-vision-API-endpoint-for-images-classification-using-ResNet50/assets/92921252/32f44a24-e3dc-4435-80b0-587587858687


## Usage

1. `pip install -r requirements.txt`

2. Run `main.py` this will run uvicorn server with the endpoints.

3. Open browser and navigate to the follwoing localhost for swagger ui `http://127.0.0.1:8000/docs`.
