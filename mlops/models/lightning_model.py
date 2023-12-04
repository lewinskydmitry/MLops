from torch import optim, nn
import lightning as L
    

class Lightning_classifier(L.LightningModule):
    def __init__(self, num_features, init_param):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(num_features, init_param),
            nn.BatchNorm1d(init_param),
            nn.ReLU(),
            nn.Linear(init_param, int(init_param / 32)),
            nn.BatchNorm1d(int(init_param / 32)),
            nn.ReLU(),
            nn.Linear(int(init_param / 32), 2),
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        result = self.classifier(x)
        loss = nn.CrossEntropyLoss()(result, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        result = self.classifier(x)
        loss = nn.CrossEntropyLoss()(result, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
