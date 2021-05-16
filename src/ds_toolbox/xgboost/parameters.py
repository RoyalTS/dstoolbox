"""xgboost-related parameters."""

default_parameters = {
    "n_estimators": 100,
    "learning_rate": 0.3,
    "gamma": 0,
    "max_depth": 6,
    "min_child_weight": 1,
    "max_delta_step": 0,
    "subsample": 1,
    "colsample_bytree": 1,
    "colsample_bylevel": 1,
    "colsample_bynode": 1,
    "reg_lambda": 1,
    "reg_alpha": 0,
    "tree_method": "auto",
    "sketch_eps": 0.03,
    "scale_pos_weight": 1,
}

native_to_sklearn_parameter_names = {
    "eta": "learning_rate",
    "lambda": "reg_lambda",
    "alpha": "reg_alpha",
}

hyperparameter_tuning_sets = {
    "learning_rate": [0.3, 0.1, 0.05, 0.01, 0.005, 0.001],
    "max_depth": [3, 4, 5, 6, 7],
    "min_child_weight": [0.5, 1, 2, 5, 10, 20],
    "subsample": [1, 0.8, 0.6],
    "colsample_bytree": [1, 0.8, 0.6, 0.4],
    "colsample_bylevel": [1, 0.8, 0.6, 0.4],
    # tree pruning minimal loss reduction
    "gamma": [0, 1, 2, 5, 10, 50, 100],
    # L1 regularization
    "reg_alpha": [0, 0.1, 1, 10, 100],
    # L2 regularization
    "reg_lambda": [0, 0.1, 1, 10, 100],
}
