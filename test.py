from torchvision.models import resnet18

model = resnet18()

print(list(model.state_dict().keys()))
