import pandas as pd
import numpy as np
from util import timeit, log, Config

F32_DATASET_SIZE = 500
F16_DATASET_SIZE = 1000


@timeit
def read_df(csv_path: str, config: Config) -> pd.DataFrame:
    if "dtype" not in config:
        preview_df(csv_path, config)

    return pandas_read_csv(csv_path, config)


@timeit
def pandas_read_csv(csv_path: str, config: Config) -> pd.DataFrame:
    return pd.read_csv(csv_path, encoding="utf-8", low_memory=False, dtype=config["dtype"], parse_dates=config["parse_dates"])


@timeit
def preview_df(train_csv: str, config: Config, nrows: int=3000):
    num_rows = sum(1 for line in open(train_csv)) - 1
    log("Rows in train: {}".format(num_rows))

    df = pd.read_csv(train_csv, encoding="utf-8", low_memory=False, nrows=nrows)
    mem_per_row = df.memory_usage(deep=True).sum() / nrows
    log("Memory per row: {:0.2f} Kb".format(mem_per_row / 1024))

    df_size = (num_rows * mem_per_row) / 1024 / 1024
    log("Approximate dataset size: {:0.2f} Mb".format(df_size))

    if df_size >= F16_DATASET_SIZE:
        float_type = np.float16
    elif df_size >= F32_DATASET_SIZE:
        float_type = np.float32
    else:
        float_type = np.float64

    config["float_type"] = float_type
    config["parse_dates"] = []
    config["dtype"] = {
        "line_id": int,
    }

    counters = {
        "number": 0,
        "string": 0,
        "datetime": 0,
    }

    for c in df:
        if c.startswith("number_") or c.startswith("id_"):
            counters["number"] += 1
            config["dtype"][c] = float_type
        elif c.startswith("string_"):
            counters["string"] += 1
            config["dtype"][c] = str
        elif c.startswith("datetime_"):
            counters["datetime"] += 1
            config["dtype"][c] = str
            config["parse_dates"].append(c)

    log("Number columns: {}".format(counters["number"]))
    log("String columns: {}".format(counters["string"]))
    log("Datetime columns: {}".format(counters["datetime"]))
