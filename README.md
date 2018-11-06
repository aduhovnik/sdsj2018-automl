## [SDSJ 2018](https://sdsj.sberbank.ai/ru/contest): AutoML Contest, TOP 10(TBA) place solution

Competition github: [sberbank-ai](https://github.com/sberbank-ai/sdsj2018-automl)

Public datasets for local validation: [sdsj2018_automl_check_datasets.zip](https://s3.eu-central-1.amazonaws.com/sdsj2018-automl/public/sdsj2018_automl_check_datasets.zip)

#### Docker :whale: 
`docker pull sberbank/python`
## This solution based on the following open solutions, thanks a lot [@vlarine](https://github.com/vlarine) and [@tyz910](https://github.com/tyz910)
- [LightGBM Baseline](https://github.com/vlarine/sdsj2018_lightgbm_baseline)
- [Docker-friendly baseline](https://github.com/tyz910/sdsj2018)

## Solution description
- Preprocessing
    - Drop constant columns
    - [Categorical encoding](https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features)
    - Datetime features to (year, weekday, month, day)
- Using LightGBM as a champion among all models and universal parameters set from [LightGBM Baseline](https://github.com/vlarine/sdsj2018_lightgbm_baseline) solution
- Check, whether ensemble can be used or not
- Estimate hyperparams
    - The most influential params are: num_leaves, min_child_weight
    - Estimation performed on holdout part of train using 20 iterations(n_estimators) with a bit more sensitive params
    - For high num_leaves values using more n_estimators
- If ensemble is used
    - use some augmented params, based on estimated params
    - get predictions by simple averaging
- Clipping to zero, if zero is a lower bound in train target
- Always think about time limit

## What I've tried but isn't presented in final solution
- Other models
    - Catboost (didn't achieve LightGBM results)
    - GLM (didn't achieve LightGBM results)
    - XGBoost (too slow)
- H2O AutoML
- Feature generation
    - Take 10 features with the lowest/highest correlation with target and use their multiplication as new features
    - Use any clustering algorithm, use clusters as new features
    - Time lags and information about holidays
- Feature selection (results were too unstable)
    - Based on features correlations
    - Based on p-values from OLS
    - [Boruta](https://github.com/scikit-learn-contrib/boruta_py) on LightGBM feature importance
- [hyperopt](https://github.com/hyperopt/hyperopt) and [BayesianOptimization](https://github.com/fmfn/BayesianOptimization)
- Estimate bias of the target for regression task, this gave me some boost on local validation
- Estimate other LightGBM params like I do for num_leaves and min_child_weight
- Leakage by [bagxi](https://github.com/bagxi/sdsj2018-leakage) :)