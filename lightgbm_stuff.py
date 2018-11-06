from util import timeit, log, Config


def get_default_params(config: Config):
    params = {
        "objective": "regression" if config["mode"][0] == "r" else "binary",
        "metric": "rmse" if config["mode"][0] == "r" else "auc",
        "learning_rate": 0.01,
        "num_leaves": 200,
        "feature_fraction": 0.70,
        "bagging_fraction": 0.70,
        "bagging_freq": 4,
        "max_depth": -1,
        "verbosity": -1,
        "reg_alpha": 0.3,
        "reg_lambda": 0.1,
        "min_child_weight": 10,
        "zero_as_missing": True,
        "seed": 18100,
    }
    return params


def get_sensitive_and_impervious_params(config: Config):
    objective = "regression" if config["mode"] == "regression" else "binary"
    params_impervious = {
        "objective": objective,
        "metric": 'rmse',
        "learning_rate": 0.01,
        "num_leaves": 200,
        "feature_fraction": 0.70,
        "bagging_fraction": 0.70,
        "bagging_freq": 4,
        "max_depth": -1,
        "verbosity": -1,
        "reg_alpha": 0.3,
        "reg_lambda": 0.1,
        "min_child_weight": 10,
        "zero_as_missing": True,
        "seed": 18127887631,
    }
    param_sensitive = {
        "learning_rate": 0.06,
        "num_leaves": 200,
        "feature_fraction": 0.95,
        "bagging_fraction": 0.95,
        "bagging_freq": 4,
        "max_depth": -1,
        "verbosity": -1,
        "reg_alpha": 0.3,
        "reg_lambda": 30,
        'n_estimators': 600,
        "min_child_weight": 1,
        "zero_as_missing": True,
        "seed": 181231,
        'objective': objective,
        'metric': 'rmse',
    }
    return [param_sensitive, params_impervious]


def get_lgbm_augmented_params(to_update: dict, metric: str, config: Config):
    params_list = []
    # first
    params = {
        "objective": "regression" if config["mode"] == "regression" else "binary",
        "metric": metric,
        "learning_rate": 0.01,
        "num_leaves": 200,
        "feature_fraction": 0.70,
        "bagging_fraction": 0.70,
        "bagging_freq": 4,
        "max_depth": -1,
        "verbosity": -1,
        "reg_alpha": 0.3,
        "reg_lambda": 0.1,
        "min_child_weight": 10,
        "zero_as_missing": True,
        "seed": 18127887631,
    }
    params.update(to_update)
    params_list.append(params)
    # second
    params = {'task': 'train',
              'boosting_type': 'gbdt',
              "objective": "regression" if config["mode"] == "regression" else "binary",
              "metric": metric,
              "learning_rate": 0.007,
              "num_leaves": 200,
              "feature_fraction": 0.72,
              "bagging_fraction": 0.72,
              'bagging_freq': 6,
              "max_depth": -1,
              "verbosity": -1,
              "reg_alpha": 0.3,
              "reg_lambda": 0.1,
              'zero_as_missing': True,
              'num_threads': 4,
              'seed': 231}
    params.update(to_update)
    params_list.append(params)
    # third
    params = {'task': 'train',
              'boosting_type': 'gbdt',
              "objective": "regression" if config["mode"] == "regression" else "binary",
              "metric": metric,
              "learning_rate": 0.013,
              "num_leaves": 200,
              "feature_fraction": 0.68,
              "bagging_fraction": 0.68,
              'bagging_freq': 4,
              "max_depth": -1,
              "verbosity": -1,
              "reg_alpha": 0.3,
              "reg_lambda": 0.1,
              'zero_as_missing': True,
              'num_threads': 4,
              'seed': 4231}
    params.update(to_update)
    params_list.append(params)
    # forth
    params = {'task': 'train',
              'boosting_type': 'gbdt',
              "objective": "regression" if config["mode"] == "regression" else "binary",
              "metric": metric,
              "learning_rate": 0.0115,
              "num_leaves": 200,
              "feature_fraction": 0.75,
              "bagging_fraction": 0.75,
              'bagging_freq': 3,
              "max_depth": -1,
              "verbosity": -1,
              "reg_alpha": 0.3,
              "reg_lambda": 0.1,
              'zero_as_missing': True,
              'num_threads': 4,
              'seed': 1}
    params.update(to_update)
    params_list.append(params)
    # fifth
    params = {'task': 'train',
              'boosting_type': 'gbdt',
              "objective": "regression" if config["mode"] == "regression" else "binary",
              "metric": metric,
              "learning_rate": 0.014,
              "num_leaves": 200,
              "feature_fraction": 0.72,
              "bagging_fraction": 0.72,
              'bagging_freq': 3,
              "max_depth": -1,
              "verbosity": -1,
              "reg_alpha": 0.3,
              "reg_lambda": 0.1,
              'zero_as_missing': True,
              'num_threads': 4,
              'seed': 31}
    params.update(to_update)
    params_list.append(params)
    # sixth
    params = {'task': 'train',
              'boosting_type': 'gbdt',
              "objective": "regression" if config["mode"] == "regression" else "binary",
              "metric": metric,
              "learning_rate": 0.0075,
              "num_leaves": 200,
              "feature_fraction": 0.7,
              "bagging_fraction": 0.7,
              'bagging_freq': 3,
              "max_depth": -1,
              "verbosity": -1,
              "reg_alpha": 0.3,
              "reg_lambda": 0.1,
              'zero_as_missing': True,
              'num_threads': 4,
              'seed': 49871}
    params.update(to_update)
    params_list.append(params)
    # seventh
    params = {'task': 'train',
              'boosting_type': 'gbdt',
              "objective": "regression" if config["mode"] == "regression" else "binary",
              "metric": metric,
              "learning_rate": 0.013,
              "num_leaves": 200,
              "feature_fraction": 0.75,
              "bagging_fraction": 0.75,
              'bagging_freq': 3,
              "max_depth": -1,
              "verbosity": -1,
              "reg_alpha": 0.3,
              "reg_lambda": 0.1,
              'zero_as_missing': True,
              'num_threads': 4,
              'seed': 523}
    params.update(to_update)
    params_list.append(params)
    return params_list
