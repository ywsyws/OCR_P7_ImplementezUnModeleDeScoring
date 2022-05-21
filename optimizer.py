from sklearnex import patch_sklearn

patch_sklearn()

import joblib
import os
import time
from typing import Tuple, Union

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import NearMiss
from lightgbm import LGBMClassifier
from optuna.samplers import TPESampler
from optuna.trial import Trial
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class OptOptuna:
    def __init__(
        self,
        config,
        scalar: str,
        custom_score: dict,
        optimizer: dict,
        features_reductions: str,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        col_keep: list,
        sampling: str = None,
    ) -> None:
        self.config = config
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.sampling = sampling
        self.features_reductions = features_reductions
        self.customscore = custom_score
        self.optimizer = optimizer

        # set scalar
        if scalar == "standard_scalar":
            self.col_transformer = ColumnTransformer(
                [("scalar", StandardScaler(), col_keep)], remainder="passthrough"
            )
        elif scalar == "minmax_scalar":
            self.col_transformer = ColumnTransformer(
                [("scalar", MinMaxScaler(), col_keep)], remainder="passthrough"
            )
        else:
            print(
                f"method has to be either 'standard_scalar' or 'minmax_scalar'. But got {scalar}."
            )

        # set model
        self.model_name = config["Model"]["name"]
        if self.model_name == "LogisticRegression":
            self.model = LogisticRegression(
                n_jobs=-1, random_state=42, verbose=2, warm_start=True, max_iter=100
            )
        elif self.model_name == "LGBMClassifier":
            self.model = LGBMClassifier(n_jobs=-1, verbose=2)
        elif self.model_name == "RandomForestClassifier":
            self.model = RandomForestClassifier(n_jobs=-1, random_state=42, verbose=2)
        elif self.model_name == "SGDClassifier":
            self.model = SGDClassifier(
                n_jobs=-1, random_state=42, verbose=2, early_stopping=True
            )

        # set sampling method
        if self.sampling == "SMOTE":
            self.resample = SMOTE(
                sampling_strategy="minority", random_state=42, n_jobs=-1
            )
        elif self.sampling == "NearMiss":
            self.resample = NearMiss(sampling_strategy="auto", n_jobs=-1)
        elif self.sampling != "ClassWeight":
            print(f"{self.sampling} is an invalide sampling method")

        # set feature selection method
        if self.features_reductions == "SelectKBest":
            self.features_reduction = SelectKBest(score_func=f_classif, k=100)

        # load study object if exists to continue the search. Otherwise, create one
        self.file_study = f"results/optimization/study_{config['Model']['shortname']}_{(self.sampling).lower()}_{(self.features_reductions).lower()}.pkl"
        if os.path.isfile(self.file_study):
            self.study = joblib.load(self.file_study)
        else:
            self.study = optuna.create_study(direction="minimize", sampler=TPESampler())

    def objective(self, trial: Trial) -> float:
        """Define an objective runction to be maximized."""

        # save study results
        os.makedirs(os.path.dirname(self.file_study), exist_ok=True)
        joblib.dump(self.study, self.file_study)

        # define parameters and set up pipeline steps
        params, steps = self.get_param_and_steps(trial)
        # set up pipeline and its parameters
        self.pipeline = Pipeline(steps=steps)
        self.pipeline.set_params(**params)

        my_score = make_scorer(self.scoring, greater_is_better=True)

        # define cross validator
        cv = RepeatedStratifiedKFold(
            n_splits=self.optimizer["n_splits"],
            n_repeats=self.optimizer["n_repeats"],
            random_state=42,
        )

        # return study results
        # set early stopping for LGBMClassifier
        if self.model_name == "LGBMClassifier":
            return np.mean(
                cross_val_score(
                    self.pipeline,
                    self.X_train,
                    self.y_train,
                    scoring=my_score,
                    error_score="raise",
                    cv=cv,
                    verbose=2,
                    fit_params={
                        "model__eval_set": [(self.X_test, self.y_test)],
                        "model__callbacks": [
                            lgb.early_stopping(stopping_rounds=200),
                            lgb.log_evaluation(200),
                        ],
                    },
                )
            )
        else:
            return np.mean(
                cross_val_score(
                    self.pipeline,
                    self.X_train,
                    self.y_train,
                    scoring=my_score,
                    cv=cv,
                    verbose=2,
                )
            )

    def opt_optuna(self) -> Tuple[np.ndarray, np.ndarray]:
        # optimize the objective function
        self.study.optimize(self.objective, n_trials=self.optimizer["n_trials"])
        # get the best parameters and set it to the pipeline
        if self.sampling == "ClassWeight":
            best_params = self.modify_best_params()
            self.pipeline.set_params(**best_params)
        else:
            self.pipeline.set_params(**self.study.best_params)

        # train model with the best parameters
        start = time.time()
        self.pipeline.fit(self.X_train, self.y_train)
        filename = f"results/models/{self.config['Model']['shortname']}_{(self.sampling).lower()}_{(self.features_reductions).lower()}.pkl"
        joblib.dump(self.pipeline, filename)

        return (
            self.pipeline.predict(self.X_test),
            self.pipeline.predict_proba(self.X_test),
            time.time() - start,
        )

    def scoring(
        self,
        y_true: Union[np.array, pd.Series],
        y_pred: Union[np.array, pd.Series],
        **kwargs,
    ) -> float:
        """
        Create custom metric for the cross validation score

        Parameters
        ----------
        y_true, y_pred: np.array or pd.Series
            The true value and prediction

        Returns
        -------
        Custom metric that emphasizes more on TP and penalizes FP
        """
        TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
        specificity = TN / (TN + FP)
        recall = TP / (TP + FN)
        return 1 - (
            self.customscore["recall_weight"] * recall
            + self.customscore["spec_weight"] * specificity
        )

    def get_min_max(self, param: str) -> Union[int, float]:
        """
        Get the minimum and maximum values of each model parameter

        Parameters
        ----------
        param: str
            The name of the parameter

        Returns
        -------
        Minimum and maximum values of the input model parameter
        """
        return (
            self.config["Model"]["params"][param]["min"],
            self.config["Model"]["params"][param]["max"],
        )

    def steps_with_resample(self) -> list:
        """
        Set up steps for the pipeline, including resampling.

        Returns
        -------
        A list of steps - 3 or 4 steps
        """
        if self.features_reductions != "NA":
            return [
                ("column_transformer", self.col_transformer),
                ("feature_selection", self.features_reduction),
                ("resample", self.resample),
                ("model", self.model),
            ]
        else:
            return [
                ("column_transformer", self.col_transformer),
                ("resample", self.resample),
                ("model", self.model),
            ]

    def steps_without_resample(self) -> list:
        """
        Set up steps for the pipeline, without resampling.

        Returns
        -------
        A list of steps -  2 or 3 steps
        """
        if self.features_reductions != "NA":
            return [
                ("column_transformer", self.col_transformer),
                ("feature_selection", self.features_reduction),
                ("model", self.model),
            ]
        else:
            return [
                ("column_transformer", self.col_transformer),
                ("model", self.model),
            ]

    def get_param_and_steps(self, trial: Trial) -> Tuple[dict, list]:
        """
        Set up parameters dictionary and pipeline steps
        according to model and resampling method.

        Returns
        -------
        Parameters dictionary
        Pipeline steps
        """
        params = {}
        for param in self.config["Model"]["params"].keys():
            param_key = "model__" + param
            if self.config["Model"]["params"][param]["distribution"] == "int":
                param_min, param_max = self.get_min_max(param)
                params[param_key] = trial.suggest_int(param_key, param_min, param_max)
            elif self.config["Model"]["params"][param]["distribution"] == "uniform":
                param_min, param_max = self.get_min_max(param)
                params[param_key] = trial.suggest_uniform(
                    param_key, param_min, param_max
                )
            elif self.config["Model"]["params"][param]["distribution"] == "loguniform":
                param_min, param_max = self.get_min_max(param)
                params[param_key] = trial.suggest_loguniform(
                    param_key,
                    param_min,
                    param_max,
                )
            elif self.config["Model"]["params"][param]["distribution"] == "categorical":
                param_list = self.config["Model"]["params"][param]["list"]
                params[param_key] = trial.suggest_categorical(param_key, param_list)

        if self.sampling == "SMOTE":
            params["resample__k_neighbors"] = trial.suggest_int(
                "resample__k_neighbors",
                self.optimizer["Smote"]["min"],
                self.optimizer["Smote"]["max"],
            )
            return params, self.steps_with_resample()
        elif self.sampling == "NearMiss":
            params["resample__n_neighbors"] = trial.suggest_int(
                "resample__n_neighbors",
                self.optimizer["NearMiss"]["min"],
                self.optimizer["NearMiss"]["max"],
            )
            return params, self.steps_with_resample()
        elif self.sampling == "ClassWeight":
            weights = trial.suggest_loguniform("model__class_weight", 0.05, 0.95)
            params["model__class_weight"] = {0: weights, 1: 1 - weights}
            return params, self.steps_without_resample()
        else:
            return params, self.steps_without_resample()

    def modify_best_params(self) -> dict:
        """
        Modify class_weight parameter of the best parameters

        Returns
        -------
        Best parameters dictionary
        """
        best_params = self.study.best_params.copy()
        class_weight = self.study.best_params["model__class_weight"]
        best_params["model__class_weight"] = {
            0: class_weight,
            1: 1 - class_weight,
        }
        return best_params
