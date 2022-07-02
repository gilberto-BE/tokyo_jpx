import torch
import torch.nn as nn
torch.manual_seed(0)

# import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchmetrics as TM
# pl.utilities.seed.seed_everything(seed=42)
from torch import nn, Tensor
import math
from metrics.metrics import compute_metrics
import matplotlib.pyplot as plt
import numpy as np


class Trainer:
    def __init__(
        self, 
        model: nn.Module, 
        optimizer_name:str='adam', 
        lr: float = 3e-6, 
        loss_fn_name: str = 'mse',
        weight_decay: float = 0.0,
        scale_ts: bool = False,

        ):

        """
        TODO
        1) Add early stopping
        """
        self.lr = lr
        self.optimizer_name=optimizer_name
        self.loss_fn_name = loss_fn_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device}-device")
        self.model = model.to(self.device)
        self.weight_decay = weight_decay
        self.scale_ts = scale_ts

        if self.loss_fn_name == 'mse':
            self.loss_fn = nn.MSELoss()

        if self.optimizer_name.lower() == 'rmsprop': 
            self.optimizer = torch.optim.RMSprop(
                self.model.parameters(), 
                self.lr, 
                weight_decay=self.weight_decay
                )

        elif self.optimizer_name.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                self.lr, 
                weight_decay=self.weight_decay
                )

    def fit_epochs(
        self, 
        train_loader: torch.utils.data.DataLoader, 
        valid_loader: torch.utils.data.DataLoader = None, 
        use_cyclic_lr: bool = False, 
        epochs: int = 5, 
        x_cat=None
        ):
        train_loss = []
        valid_loss = []
        train_mae = []
        valid_mae = []
        best_valid_loss = 1_000_000
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            'min',
            factor=0.5, 
            patience=5, 
            threshold=1e-3, 
            verbose=True
            )
        for epoch in range(epochs):
            result = self.fit_one_epoch(train_loader, valid_loader, use_cyclic_lr, x_cat=x_cat)
            scheduler.step(result['avg_loss_val'])
            if epoch % 10  == 0:
                print(
                    f"""
                    Average train loss: {result["avg_loss_train"]} | 
                    Train-Mae: {result["train_mae"]} |

                    Average val loss: {result["avg_loss_val"]}|
                    Val-Mae: {result["val_mae"]}
                    """
                    )
            if result['avg_loss_val'] < best_valid_loss:
                """
                SAVE THE BEST MODEL BASED ON BEST VALID LOSS
                """
                pass
            train_loss.append(result["avg_loss_train"])
            valid_loss.append(result["avg_loss_val"].cpu().detach().numpy())
            train_mae.append(result["train_mae"])
            valid_mae.append(result["val_mae"])
        return train_loss, train_mae, valid_loss, valid_mae

    def fit_one_epoch(
        self, 
        train_loader: torch.utils.data.DataLoader, 
        valid_loader:torch.utils.data.DataLoader=None, 
        use_cyclic_lr:bool=False, 
        x_cat:bool=None
        ):
        if use_cyclic_lr:
            """add parameters for scheduler to constructor."""
        self.model.train()
        pred_train, avg_loss_train, train_metrics = self.run_train_step(train_loader, x_cat=x_cat)

        if valid_loader is not None:
            pred_val, avg_loss_val, val_metrics = self.run_val_step(valid_loader, x_cat=x_cat)

        result = {
            'pred_train': pred_train, 
            'avg_loss_train': avg_loss_train, 
            'pred_val': pred_val, 
            'avg_loss_val': avg_loss_val,
            'train_mae': train_metrics['mae'],
            'val_mae': val_metrics['mae']
            }
        return result

    def run_train_step(self, train_loader, loss_every=100, x_cat=True):
        running_loss = 0.0
        last_loss = 0.0
        for batch, data in enumerate(train_loader):
            x = data['num_features'].to(self.device)
            y = data['target'].to(self.device)
            self.optimizer.zero_grad()

            """
            TODO: CHECK SCALING ISSUES
            USE MAX-SCALING???
            """
            # if self.scale_ts:
            #     scaling_factor = torch.abs(torch.max(x))
            #     x = x/scaling_factor
            if x_cat is not None:
                x_cat = data['cat_features'].to(self.device)
                pred = self.model(x, x_cat).to(self.device)
            else:
                pred = self.model(x).to(self.device)

            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            if batch % loss_every == 0:
                last_loss = running_loss/loss_every
                running_loss = 0.0

        train_metrics = compute_metrics(pred, y)
        return pred, last_loss, train_metrics

    def run_val_step(self, valid_loader, x_cat=True):
        running_loss = 0.0
        self.model.eval()
        # with torch.no_grad:
        for batch, data in enumerate(valid_loader):
            x = data['num_features'].to(self.device)
            y = data['target'].to(self.device)
            # if self.scale_ts:
            #     scaling_factor = torch.abs(torch.max(x))
            #     x = x/scaling_factor
            if x_cat is not None:
                x_cat = data['cat_features'].to(self.device)
                pred = self.model(x, x_cat).to(self.device)
            else:
                pred = self.model(x)

            loss = self.loss_fn(pred, y)
            running_loss += loss
        avg_loss = running_loss/(batch + 1)
        val_metrics = compute_metrics(pred, y)
        return pred, avg_loss, val_metrics

    def save_model(self, model, path='./trained_model.pt'):
        torch.save(model, path)

    def load_model(self, path='./notebooks/trained_model.pt'):
        model = torch.load(path)
        return model.eval()


def plot_loss(line1, line2, title1='Train-MAE', title2='Valid-MAE'):
    fig, ax = plt.subplots()
    training_mae, = ax.plot(range(len(line1)), line1, label=title1)
    val_mae, = ax.plot(range(len(line2)), line2, label=title2)
    plt.xlabel('Epochs')
    ax.legend(handles=[training_mae, val_mae])
    plt.show()

