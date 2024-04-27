import torch
import torchvision

def convert_and_download():

    # Load pretrained ResNet50 model
    model = torchvision.models.resnet50(pretrained=True)

    # Use torch.onnx.export to convert the model to ONNX format
    torch.onnx.export(model,               # model being run
                    input,                         # model input (or a tuple for multiple inputs)
                    "resnet50.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=12,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})
