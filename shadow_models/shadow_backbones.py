import torch
import torch.nn as nn
import torch.nn.functional as F

available_shadow_backbones = ["simplecnn"]

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, input_features=3):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=input_features, out_channels=6, padding=0, kernel_size=5, stride=1),
                nn.MaxPool2d(kernel_size=2),
                nn.Tanh(),
            )

        self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=6, out_channels=16, padding=0, kernel_size=5, stride=1),
                nn.MaxPool2d(kernel_size=2),
                nn.Tanh(),
            )

        self.fc1 = nn.Sequential(
                nn.Linear(in_features=16*5*5, out_features=128, bias=True),
                nn.Tanh(),
            )

        self.fc2 = nn.Sequential(
                nn.Linear(in_features=128, out_features=10, bias=True),
                nn.Sigmoid(),
            )

        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        #x = self.softmax(x)
        return x

def get_shadow_model(model, num_classes=10, input_features=3, checkpoint_path=None, device=torch.device("cpu")):
    assert model in available_shadow_backbones, "You have specified a shadow model that has not been implemented. Please choose a model from " + str(available_shadow_backbones)

    if model == "simplecnn":
        model = SimpleCNN(num_classes=num_classes, input_features=input_features)
    
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        try:
            model.load_state_dict(checkpoint["state_dict"])
        except KeyError:
            model.load_state_dict(checkpoint)
        
    model.to(device)
    return model