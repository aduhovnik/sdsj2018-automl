import pandas as pd
import numpy as np
import time
from automl import AutoML
from util import timeit, log
import mlflow

# it's important to place your mlflow directory and data out of your project
mlflow.set_tracking_uri('../mlruns')
mlflow.set_experiment('sdsj2018')

DATA_PATH = '../sdsj_data'

DATASETS = [
    ("1", "regression"),
    ("2", "regression"),
    ("3", "regression"),
    ("4", "classification"),
    ("5", "classification"),
    ("6", "classification"),
    ("7", "classification"),
    ("8", "classification"),
]


@timeit
def validate_dataset(alias: str, mode: str) -> (np.float64, np.float64, np.float64):
    log(alias)
    automl = AutoML("models/check_{}".format(alias))
    automl.train(DATA_PATH + "/check_{}/train.csv".format(alias), mode)
    _, score = automl.predict(DATA_PATH + "/check_{}/test.csv".format(alias), "predictions/check_{}.csv".format(alias))
    return score


if __name__ == '__main__':
    scores = {
        "dataset": [],
        "score": [],
        "time": [],
    }
    with mlflow.start_run():
        for i, mode in DATASETS:
            alias = "{}_{}".format(i, mode[0])

            start_time = time.time()
            score = validate_dataset(alias, mode)
            end_time = time.time()

            scores["dataset"].append(alias)
            scores["score"].append(score)
            scores["time"].append(end_time - start_time)

            alias = alias[:-2]
            mlflow.log_metric('score_{}'.format(alias), score)

        scores = pd.DataFrame(scores)
        scores.to_csv("scores/{}.csv".format(int(time.time())))
        mlflow.log_artifacts('./')
        print(scores)
