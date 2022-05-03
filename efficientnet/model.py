import torch
import timm


class WireClassifier(torch.nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(WireClassifier, self).__init__()
        self.model = timm.create_model(
            "efficientnet_b4", pretrained=pretrained, num_classes=num_classes
        )
        # self.softmax = torch.nn.Softmax(dim=1)

        # self.model.head.fc = torch.nn.Linear(self.model.head.fc.in_features, args.num_classes)

    def forward(self, x):
        x = self.model(x)
        # x = self.softmax(x)
        return x
