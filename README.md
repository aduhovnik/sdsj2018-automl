## [SDSJ 2018](https://sdsj.sberbank.ai/ru/contest): AutoML Contest, TOP 10(TBA) place solution

Competition github: [sberbank-ai](https://github.com/sberbank-ai/sdsj2018-automl)

Public datasets for local validation: [sdsj2018_automl_check_datasets.zip](https://s3.eu-central-1.amazonaws.com/sdsj2018-automl/public/sdsj2018_automl_check_datasets.zip)

#### Docker :whale: 
`docker pull sberbank/python`
## This solution based on the following open solutions, thanks a lot [@vlarine](https://github.com/vlarine) and [@tyz910](https://github.com/tyz910)
- [LightGBM Baseline](https://github.com/vlarine/sdsj2018_lightgbm_baseline)
- [Docker-friendly baseline](https://github.com/tyz910/sdsj2018)

## Solution description:
1) Preprocessing: 
    1) fill nan
    2) drop constant columns
    3) [categorical encoding](https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features)
    4) datetime features to (year, weekday, month, day)
2) Using LightGBM as a champion among all models and universal parameters set from [LightGBM Baseline](https://github.com/vlarine/sdsj2018_lightgbm_baseline) solution
3) Check, whether ensemble can be used or not
4) Estimate hyperparams
    1) the most wanted params are: num_leaves, min_child_weight
    2) estimation performed on holdout part of train using 20 iterations(n_estimators) with a bit more sensitive params
    3) for high num_leaves values using more n_estimators
5) If ensemble is used, than use some augmented params, based on estimated params
6) If ensemble is used, then get predictions by simple averaging
7) Clipping to zero, if zero is a lower bound in train target
8) Always think about time limit

## What I've tried but isn't presented in final solution
1) Other models
    1) Catboost (didn't achieve LightGBM results)
    2) GLM (didn't achieve LightGBM results)
    3) XGBoost (too slow)
2) H2O AutoML
3) Feature generation:
    1) Take 10 features with the lowest/highest correlation with target and use their multiplication as new features
    2) Use any clustering algorithm, use clusters as new features
    3) Time lags and information about holidays
4) Feature selection: (results were too unstable)
    1) Based on features correlations
    2) Based on p-values from OLS
    3) [Boruta](https://github.com/scikit-learn-contrib/boruta_py) on LightGBM feature importance
5) [hyperopt](https://github.com/hyperopt/hyperopt) and [BayesianOptimization](https://github.com/fmfn/BayesianOptimization)
6) Leakage by [bagxi](https://github.com/bagxi/sdsj2018-leakage) :)