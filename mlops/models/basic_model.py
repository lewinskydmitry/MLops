import torch.nn as nn


class Baseline_classifier(nn.Module):
    def __init__(self, num_features, init_param):
        super(Baseline_classifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(num_features, init_param),
            nn.BatchNorm1d(init_param),
            nn.ReLU(),
            nn.Linear(init_param, int(init_param / 32)),
            nn.BatchNorm1d(int(init_param / 32)),
            nn.ReLU(),
            nn.Linear(int(init_param / 32), 2),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x
