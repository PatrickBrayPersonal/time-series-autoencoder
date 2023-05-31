import hydra
import mlflow
import torch
import torch.nn as nn
from hydra.utils import instantiate

from tsa import AutoEncForecast, evaluate, train
from tsa.infer import infer
from tsa.utils import load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(config_path="./", config_name="config")
def run(cfg):
    ts = instantiate(cfg.data)
    infer_iter = ts.get_infer_loader()
    mlflow.set_tracking_uri(
        "file://" + hydra.utils.get_original_cwd() + "/mlruns"
    )
    model = mlflow.pytorch.load_model(f"models:/{cfg.model.name}/{1}")
    infer(infer_iter, model)
    print("complete")


if __name__ == "__main__":
    run()
