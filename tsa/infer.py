import mlflow
import numpy as np
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer(inf_iter, model):
    """
    Evaluate the model on the given test set.

    Args:
        test_iter: (DataLoader): test dataset iterator
        criterion: loss function
        model: model to use
        config: config
    """
    predictions, errors = [], []

    model.eval()
    errors = []
    for batch in tqdm(inf_iter, total=len(inf_iter), desc="Evaluating"):
        with torch.no_grad():
            feature, y_hist, target = batch
            output, att = model(
                feature.to(device), y_hist.to(device), return_attention=True
            )
            predictions.append(output.squeeze(1).cpu())
            errors.append((output - target).numpy())
    return np.concatenate(errors)
