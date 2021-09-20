import torch
import torch.nn as nn

available_attack_backbones = ["fcnet"]

class FCNet(nn.Module):
    def __init__(self, in_features, layer_dims):
        super(FCNet, self).__init__()

        self.module = nn.Sequential()
        for i, layer_dim in enumerate(layer_dims):
            self.module.add_module(
                "fc_" + str(i+1), 
                nn.Linear(in_features=in_features, out_features=layer_dim, bias=False)
            )

            self.module.add_module(
                "batchnorm_" + str(i+1), 
                nn.BatchNorm1d(num_features=layer_dim)
            )
            
            self.module.add_module(
                "relu_" + str(i+1), 
                nn.ReLU()
            )
            
            in_features = layer_dim

        self.fc_out = nn.Linear(in_features=in_features, out_features=1, bias=True)
        self.out_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.module(x)
        x = self.fc_out(x)
        x = self.out_activation(x)

        return x

def get_attack_model(model, in_features, layer_dims, checkpoint_path, device):
    assert model in available_attack_backbones, "You have selected a model that has not been implemented yet. Please select a model from " + str(available_attack_backbones)

    if model == "fcnet":
        model = FCNet(in_features=in_features, layer_dims=layer_dims)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        try:
            model.load_state_dict(checkpoint["state_dict"])
        except KeyError:
            model.load_state_dict(checkpoint)
    
    model.to(device)
    return model