import torch
import torch.nn as nn
import torch.nn.functional as F

available_victim_backbones = ["simplecnn", "pytorchcnn", "simplemlp"]

class PytorchCNN(nn.Module):
    def __init__(self, num_classes=10, input_features=3):
        super(PytorchCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

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

class SimpleMLP(nn.Module):
    def __init__(self, num_classes=10, input_features=10, layer_dims=[128]):
        super(SimpleMLP, self).__init__()
        
        self.linear_tanh_stack = nn.Sequential()


        for i, layer in enumerate(layer_dims):
            self.linear_tanh_stack.add_module(
                    "fc_" + str(i),
                    nn.Linear(in_features=input_features, out_features=layer, bias=True),
                    )
            self.linear_tanh_stack.add_module(
                    "tanh_" + str(i),
                    nn.Tanh(),
                    )

            input_features = layer

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear_tanh_stack(x)
        x = self.softmax(x)
        return x


def get_victim_model(model_name, num_classes=10, input_features=3, checkpoint_path="./victims/checkpoints/cifar10_15000-simplecnn/checkpoint.pth", 
                                device=torch.device("cpu")):
    assert model_name in available_victim_backbones, "You have specified a model that is not implemented. Please specify a model from " + str(available_victim_backbones)

    if model_name == "pytorchcnn":
        model = PytorchCNN(num_classes=num_classes, input_features=input_features)
    elif model_name == "simplecnn":
        model = SimpleCNN(num_classes=num_classes, input_features=input_features)
    elif model_name == "simplemlp":
        model = SimpleMLP(num_classes=num_classes, input_features=input_features)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        try:
            model.load_state_dict(checkpoint["state_dict"])
        except KeyError:
            model.load_state_dict(checkpoint)
    
    model.to(device)
    return model