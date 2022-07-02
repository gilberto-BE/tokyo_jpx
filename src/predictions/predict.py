import torch
import pandas as pd
import numpy as np
# from metrics import metrics


class Predict:
    def __init__(self, path_to_model):
        self.path_to_model = path_to_model
    
    def load_model(self):
        self.model = torch.load(self.path_to_model)
        return self.model.eval()


def run_pred_step(test_loader, model, x_cat=None):
    device = "cpu"
    model.eval()
    try:
        for batch, data in enumerate(test_loader):
            x = data['num_features'].to(device)
            if x.shape[0] > 0:
                if x_cat is not None:
                    x_cat = data['cat_features'].to(device)
                with torch.torch.no_grad():
                    pred = model(x, x_cat).to(device).detach().numpy()
            else:
                return np.nan
        return pred.squeeze()
    except Exception as e:
        print(f'Exception from prediciton script: {e}')
            