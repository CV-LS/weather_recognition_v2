import sys
sys.path.append('../../')
import torch
from weather_recognition.options.config import Common
# Model = ResNet([3, 4, 6, 3]).cuda()
# Model.load("/home/sby/ColorConstancy/YZheng_model/ResNet_Cube++/netG_model_epoch_24_loss_2.215341567993164.pth")
RELOAD_CHECKPOINT_PATH = "../MobileNetV4_pytorch/data_final/mobilev4_numpy_val4_0.8586251621271076.pth"
model = torch.load(RELOAD_CHECKPOINT_PATH,map_location=torch.device('cpu'))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dummy_input = torch.randn(1, 3, 448, 256,device=device)
input_names = ["input"]
output_names = ["output"]
model.to(Common.device)
# torch.onnx.export(
#     Model,
#     dummy_input,
#     # Color,
#     operator_export_type=12,
#     example_outputs= dummy_output,
    
# )
torch.onnx.export(model, dummy_input, "mobilenetv4_448x256.onnx", opset_version=12, verbose=False, output_names=["hm"])
# torch.onnx.export(Model, dummy_input, "model_.onnx", opset_version=12, verbose=False, output_names=["hm"])
