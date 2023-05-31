import hydra
import torch
import torch.nn as nn
from hydra.utils import instantiate
from pathlib import Path
import os
from tsa import AutoEncForecast, train, evaluate
from tsa.utils import load_checkpoint
import mlflow
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@hydra.main(config_path="./", config_name="config")
def run(cfg):
    print(cfg)
    mlflow.set_tracking_uri('file://' + hydra.utils.get_original_cwd() + '/mlruns')
    mlflow.set_experiment(experiment_name="1")
    ts = instantiate(cfg.data)
    train_iter, test_iter, nb_features = ts.get_loaders()

    model = AutoEncForecast(cfg.training, input_size=nb_features).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    with mlflow.start_run() as mlrun:
        if cfg.general.do_train:
            model, criterion = train(train_iter, test_iter, model, criterion, optimizer, cfg, ts)
        if cfg.general.do_eval:
            evaluate(test_iter, criterion, model, cfg, ts)
        mlflow.pytorch.log_model(model, "model")
        with open("model.p", "wb") as f:
            import pickle
            pickle.dump(model, f)
            mlflow.log_artifact("model.p")


if __name__ == "__main__":
    run()
