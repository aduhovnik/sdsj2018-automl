import datetime
import numpy as np
import pandas as pd
from util import timeit, log, Config


@timeit
def preprocess(df: pd.DataFrame, config: Config):
    set_clip_values(df, config)
    drop_columns(df)
    fillna(df, config)
    transform_datetime(df, config)
    transform_categorical(df, config)
    drop_constant_columns(df, config)


@timeit
def set_clip_values(df: pd.DataFrame, config: Config):
    if 'target' not in df.columns:
        return
    min_value = df['target'].min()
    max_value = df['target'].max()
    perc_80 = np.percentile(df['target'], 80)
    config['clip_values'] = (min_value, max_value, perc_80)


@timeit
def drop_columns(df: pd.DataFrame):
    df.drop([c for c in ["is_test", "line_id"] if c in df], axis=1, inplace=True)


@timeit
def fillna(df: pd.DataFrame, config: Config):
    for c in [c for c in df if c.startswith("number_")]:
        df[c].fillna(config["float_type"](-1), inplace=True)

    for c in [c for c in df if c.startswith("string_")]:
        df[c].fillna("", inplace=True)

    for c in [c for c in df if c.startswith("datetime_")]:
        df[c].fillna(datetime.datetime(1970, 1, 1), inplace=True)


@timeit
def drop_constant_columns(df: pd.DataFrame, config: Config):
    if "constant_columns" not in config:
        config["constant_columns"] = [c for c in df if c.startswith("number_") and not (df[c] != df[c].iloc[0]).any()]
        log("Constant columns: " + ", ".join(config["constant_columns"]))

    if len(config["constant_columns"]) > 0:
        df.drop(config["constant_columns"], axis=1, inplace=True)


@timeit
def transform_datetime(df: pd.DataFrame, config: Config):
    date_parts = ["year", "weekday", "month", "day"]

    if "date_columns" not in config:
        config["date_columns"] = {}

        for c in [c for c in df if c.startswith("datetime_")]:
            config["date_columns"][c] = []
            for part in date_parts:
                part_col = c + "_" + part
                df[part_col] = getattr(df[c].dt, part)

                if not (df[part_col] != df[part_col].iloc[0]).any():
                    log(part_col + " is constant")
                    df.drop(part_col, axis=1, inplace=True)
                else:
                    config["date_columns"][c].append(part)
            df.drop(c, axis=1, inplace=True)
    else:
        for c, parts in config["date_columns"].items():
            for part in parts:
                part_col = c + "_" + part
                df[part_col] = getattr(df[c].dt, part)
            df.drop(c, axis=1, inplace=True)


@timeit
def transform_categorical(df: pd.DataFrame, config: Config):
    # https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
    if "categorical_columns" not in config:
        min_samples_leaf = int(0.01 * len(df))
        smoothing = int(0.005 * len(df))
        prior = df["target"].mean()

        config["categorical_columns"] = {}
        for c in [c for c in df if c.startswith("string_")]:
            averages = df[[c, "target"]].groupby(c)["target"].agg(["mean", "count"])
            smooth = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
            averages["target"] = prior * (1 - smooth) + averages["mean"] * smooth
            config["categorical_columns"][c] = averages["target"].to_dict()

    for c, values in config["categorical_columns"].items():
        df.loc[:, c] = df[c].apply(lambda x: values[x] if x in values else config["float_type"](-1))
