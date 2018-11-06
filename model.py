import pandas as pd
import numpy as np
import lightgbm as lgb
from util import timeit, log, Config
from typing import List
from sklearn.metrics import mean_squared_error, roc_auc_score
from collections import defaultdict
from sklearn.model_selection import train_test_split
from lightgbm_stuff import get_default_params, get_lgbm_augmented_params, get_sensitive_and_impervious_params
import time

COLD_TOUCH_ITERATIONS = 20
NUM_LEAVES_SPACE = [20, 30, 50, 70, 80, 100, 150, 200, 250, 300]
MIN_CHILD_WEIGHT_SPACE = [5, 10, 30, 50, 100]


def get_score(truth, preds, config: Config):
    if config["mode"][0] == "c":
        return roc_auc_score(truth, preds)
    else:
        return np.sqrt(mean_squared_error(truth, preds))


@timeit
def can_we_use_ensemble(x: pd.DataFrame, config: Config):
    magic_const = 4. * 10 ** -6  # something like operations per second
    seconds_to_ensemble_train = x.shape[0] * x.shape[1] * magic_const * 5
    seconds_to_ensemble_predict = x.shape[0] * x.shape[1] * magic_const / 10
    if seconds_to_ensemble_train < config['time_limit'] and seconds_to_ensemble_predict < 300:
        return True
    else:
        return False


@timeit
def train(x: pd.DataFrame, y: pd.Series, config: Config):
    if can_we_use_ensemble(x, config):
        log("Lets do ensemble")
        train_lightgbm_ensemble(x, y, config)
    else:
        train_lightgbm(x, y, config)


@timeit
def predict(x: pd.DataFrame, config: Config) -> List:
    return predict_lightgbm(x, config)


@timeit
def estimate_value(x: pd.DataFrame, y: pd.Series, config: Config, param_name: str, values: list, to_check: dict=None):
    if not to_check:
        params_to_check = get_default_params(config)
    else:
        params_to_check = to_check

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.33, random_state=42)
    res = defaultdict(list)
    for v in values:
        params_to_check.update({param_name: v})
        model = lgb.train(params_to_check, lgb.Dataset(x_train, label=y_train), COLD_TOUCH_ITERATIONS)
        res['valid'].append(get_score(y_valid, model.predict(x_valid), config))

    chooser = np.argmax if config['mode'][0] == 'c' else np.argmin
    best_param_value = values[chooser(res['valid'])]
    log('Estimated {} value: {}'.format(param_name, best_param_value))
    return best_param_value


@timeit
def get_params_to_params_estimation(x: pd.DataFrame, y: pd.Series, config: Config):
    params_to_check = get_sensitive_and_impervious_params(config)

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.33, random_state=42)
    res = defaultdict(list)
    for p in params_to_check:
        model = lgb.train(p, lgb.Dataset(x_train, label=y_train), COLD_TOUCH_ITERATIONS)
        res['valid'].append(get_score(y_valid, model.predict(x_valid), config))

    chooser = np.argmax if config['mode'][0] == 'c' else np.argmin
    best_params = params_to_check[chooser(res['valid'])]
    log('Estimated params: {}'.format(best_params))
    return best_params


@timeit
def train_lightgbm_ensemble(x: pd.DataFrame, y: pd.Series, config: Config):
    params_to_check = get_params_to_params_estimation(x, y, config)

    leaves = estimate_value(x, y, config, 'num_leaves', NUM_LEAVES_SPACE, params_to_check)
    min_child_weight = estimate_value(x, y, config, 'min_child_weight', MIN_CHILD_WEIGHT_SPACE)

    iterations = 620
    # if too many leaves, then we can increase iterations(n_estimators), model could find more dependencies (no proof)
    if leaves > 250:
        iterations += 200

    to_update = {
        'num_leaves': leaves,
        'min_child_weight': min_child_weight
    }

    config['many_models'] = True
    config['model'] = []

    metric = "rmse" if config["mode"] == "regression" else "auc"
    lgm_params = get_lgbm_augmented_params(to_update, metric, config)
    first_set, other_params = lgm_params[0], lgm_params[1:]

    start_one_iter = time.time()
    config["model"].append(lgb.train(first_set, lgb.Dataset(x, label=y), iterations))
    duration = time.time() - start_one_iter

    for ix, lgbm_set in enumerate(lgm_params):
        if duration + time.time() + 10 <= config['start_time'] + config['time_limit']:
            log('One more model for ensemble: {}'.format(ix))
            # for higher learning_rate use lower number of iterations and vice versa, 0.01 lr is default value
            iters_delta = int(-(lgbm_set['learning_rate'] - 0.01) * 20000)
            log('Iters: {}'.format(iterations + iters_delta))
            config['model'].append(lgb.train(lgbm_set, lgb.Dataset(x, label=y), iterations+iters_delta))


@timeit
def train_lightgbm(x, y, config):
    leaves = estimate_value(x, y, config, 'num_leaves', NUM_LEAVES_SPACE)
    min_child_weight = estimate_value(x, y, config, 'min_child_weight', MIN_CHILD_WEIGHT_SPACE)

    params = get_default_params(config)
    params['num_leaves'] = leaves
    params['min_child_weight'] = min_child_weight

    config["model"] = lgb.train(params, lgb.Dataset(x, label=y), 600)
    config['many_models'] = False


@timeit
def validate(preds: pd.DataFrame, target_csv: str, mode: str) -> np.float64:
    df = pd.merge(preds, pd.read_csv(target_csv), on="line_id", left_index=True)
    score = roc_auc_score(df.target.values, df.prediction.values) if mode == "classification" else \
        np.sqrt(mean_squared_error(df.target.values, df.prediction.values))
    log("Score: {:0.4f}".format(score))
    return score


@timeit
def predict_lightgbm(x: pd.DataFrame, config: Config) -> List:
    if config['many_models']:
        predict = 0
        for model_ in config['model']:
            predict += model_.predict(x)
        predict /= len(config['model'])
    else:
        predict = config["model"].predict(x)

    if config['mode'] == 'regression':
        min_value, max_value, p80 = config['clip_values']
        if min_value == 0:
           predict = predict.clip(min=min_value)
        if max_value == p80:
           predict = predict.clip(max=max_value)
    return predict
