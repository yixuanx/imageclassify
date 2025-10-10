import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np

# 加载ImageNet类别标签
def load_imagenet_classes():
    with open('imagenet_classes.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

# 图像预处理（与ResNet训练时的预处理一致）
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet均值
        std=[0.229, 0.224, 0.225]    # ImageNet标准差
    )
])

# 加载ResNet50模型（预训练）
def load_model():
    model = models.resnet50(pretrained=True)
    model.eval()  # 切换到评估模式
    return model

# 预测函数
def predict_image(model, image, classes, top_k=3):
    # 预处理图像
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # 增加批次维度

    # 禁用梯度计算（加速推理）
    with torch.no_grad():
        output = model(input_batch)

    # 计算概率（softmax）
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # 获取top-k结果
    top_prob, top_indices = torch.topk(probabilities, top_k)
    top_classes = [classes[idx] for idx in top_indices]
    
    # 转换为列表返回
    return [
        {"class": cls, "probability": float(prob)} 
        for cls, prob in zip(top_classes, top_prob)
    ]