import torch
import torch.nn as nn
from model.utils import build_vit, build_swin


class SimCLR(nn.Module):

    def __init__(self, base_model, config, out_dim):
        super(SimCLR, self).__init__()
        self.resnet_dict = {"vit": build_vit(config),
                            "swin": None}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.embed_dim

        # add mlp projection head
        self.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                nn.ReLU(),
                                torch.nn.Dropout(0.1),
                                nn.Linear(dim_mlp, 512),
                                nn.ReLU(),
                                torch.nn.Dropout(0.1),
                                nn.Linear(512, out_dim)
                                )

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except Exception:
            print("Invalid backbone architecture. Check the config file and pass one of: Vit or Swin Transformer")
        else:
            return model

    def forward(self, x):
        out = self.backbone(x)
        return self.fc(out)
if __name__ == "__main__":
    import yaml
    from config.config import AttrDict
    yaml_file_path = "/home/jackson/Desktop/Paper/config/simmim_pretrain__vit_base__img224__800ep.yaml"
    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)
    config = AttrDict(data)
    model = SimCLR("vit", config, out_dim=256)
    model.backbone.load_state_dict(torch.load("/home/jackson/Desktop/Paper/pretrain_model/vit_base_image224_800ep.pt"))
    print(model)
