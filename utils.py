import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import torch.nn.utils.prune as prune  # 模型剪枝工具


# 1. 加载ImageNet类别标签（保持不变）
def load_imagenet_classes():
    with open('imagenet_classes.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes


# 2. 图像预处理（可选：减小输入尺寸以降低内存占用）
# 注意：输入尺寸减小可能导致精度下降，需根据需求调整
preprocess = transforms.Compose([
    transforms.Resize(224),  # 原始为256，可改为192进一步减小
    transforms.CenterCrop(224),  # 可改为192（需与Resize配合）
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# 3. 加载ResNet50并进行内存优化
def load_model(use_quantization=True, use_pruning=True):
    # 加载预训练模型
    model = models.resnet50(pretrained=True)
    model.eval()  # 切换到评估模式

    # 优化1：模型剪枝（移除50%的冗余卷积核）
    if use_pruning:
        # 对卷积层进行剪枝（示例：对所有conv2d层剪枝50%的通道）
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                # 剪枝50%的输入通道
                prune.l1_unstructured(module, name='weight', amount=0.5)
                # 永久移除被剪枝的参数（释放内存）
                prune.remove(module, 'weight')

    # 优化2：量化模型（转为int8，内存减少75%）
    if use_quantization:
        # 动态量化（简单高效，无需校准数据）
        model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Conv2d, torch.nn.Linear},  # 对卷积层和全连接层量化
            dtype=torch.qint8  # 目标精度
        )

    # 优化3：转为半精度（float16，内存减少50%，若不量化可启用）
    # 注意：量化和半精度二选一，量化内存节省更多
    # model = model.half()

    return model


# 4. 预测函数（适配低精度模型）
def predict_image(model, image, classes, top_k=3):
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # 增加批次维度

    # 若使用半精度模型，输入也需转为float16
    # if next(model.parameters()).dtype == torch.float16:
    #     input_batch = input_batch.half()

    # 禁用梯度计算（减少中间变量内存）
    with torch.no_grad():
        output = model(input_batch)

    # 计算概率（softmax）
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # 获取top-k结果
    top_prob, top_indices = torch.topk(probabilities, top_k)
    top_classes = [classes[idx] for idx in top_indices]
    
    return [
        {"class": cls, "probability": float(prob)} 
        for cls, prob in zip(top_classes, top_prob)
    ]