import itertools
import pickle
import re
import time
import yaml

import pandas as pd

from metrics import get_metrics
from optimizer import OptOptuna


def get_non_bool_col(X_train: pd.DataFrame, X_test: pd.DataFrame) -> list:
    X = pd.concat([X_train, X_test])
    return [col for col in X if X[col].nunique() != 2]


def main():
    # get config
    with open("configs/config.yml", "r") as f:
        config = yaml.safe_load(f)

    # read data
    X_train = pd.read_csv(config["Global"]["path_Xtrain"])
    y_train = pd.read_csv(config["Global"]["path_ytrain"])
    X_test = pd.read_csv(config["Global"]["path_Xtest"])
    y_test = pd.read_csv(config["Global"]["path_ytest"])
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()

    # declare variables
    classifiers = []
    samplings = []
    features_reductions = []
    preds = []
    scores = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    fprs = []
    tprs = []
    thresholds_rocs = []
    roc_aucs = []
    prec_prs = []
    recall_prs = []
    thresholds_prs = []
    conf_mat = []
    custom_scores = []
    durations = []

    search_spaces = list(
        itertools.product(
            config["Model"]["paths"],
            config["Sampling"]["method"],
            config["FeatureReduction"]["method"],
        )
    )
    # refine search spaces
    search_spaces = list(
        set(
            [
                search_space[:2]
                if (search_space[0] == "configs/config_lgbm.yml")
                or (search_space[0] == "configs/config_rf.yml")
                else search_space
                for search_space in search_spaces
            ]
        )
    )

    for idx, search_space in enumerate(search_spaces):
        start = time.time()
        # declare variables
        sampling = search_space[1]
        if len(search_space) > 2:
            features_reduction = search_space[2]
        else:
            features_reduction = "NA"
        # read model configs
        with open(search_space[0], "r") as f:
            config_model = yaml.safe_load(f)

        # rename columns name for LGBMClassifier
        if config_model["Model"]["name"] == "LGBMClassifier":
            X_train = X_train.rename(
                columns={c: re.sub("[^A-Za-z0-9_]+", "", c) for c in X_train.columns}
            )
            X_test = X_test.rename(
                columns={c: re.sub("[^A-Za-z0-9_]+", "", c) for c in X_test.columns}
            )
        else:
            X_train = X_train_copy.copy()
            X_test = X_test_copy.copy()

        # get dataframe columns to be scaled
        col_keep = get_non_bool_col(X_train, X_test)

        print(f"Starting model {idx+1} of {len(search_spaces)} models...")
        print(search_space)
        print("=========================================================")
        # instantiate optimization class
        oo = OptOptuna(
            config_model,
            config["Preprocessing"]["scalar"],
            config["CustomScore"],
            config["Optimizer"],
            features_reduction,
            X_train,
            y_train.TARGET,
            X_test,
            y_test.TARGET,
            col_keep,
            sampling,
        )
        # find best parameters and get predictions accordingly
        pred, score, duration = oo.opt_optuna()

        # append information for the metric dataframe
        cls_name = config_model["Model"]["name"]
        classifiers.append(cls_name)
        samplings.append(sampling)
        features_reductions.append(features_reduction)
        preds.append(pred)
        scores.append(score)

        # get metrics
        (
            acc,
            prec,
            recall,
            f1,
            fpr,
            tpr,
            thresholds_roc,
            roc_auc,
            prec_pr,
            recall_pr,
            thresholds_pr,
            confusion_matrix,
        ) = get_metrics(y_test, pred, score[:, 1])
        print(f"Duration of {search_space} is ", time.time() - start)

        # append metrics for the metric dataframe
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(recall)
        f1s.append(f1)
        fprs.append(fpr)
        tprs.append(tpr)
        thresholds_rocs.append(thresholds_roc)
        roc_aucs.append(roc_auc)
        prec_prs.append(prec_pr)
        recall_prs.append(recall_pr)
        thresholds_prs.append(thresholds_pr)
        conf_mat.append(confusion_matrix)
        TN, FP, _, _ = confusion_matrix.ravel()
        specificity = TN / (TN + FP)
        custom_score = 1 - (
            config["CustomScore"]["recall_weight"] * recall
            + config["CustomScore"]["spec_weight"] * specificity
        )
        custom_scores.append(custom_score)
        durations.append(duration)

        # print major metrics
        print("Accuracy:    ", acc)
        print("Precision:   ", prec)
        print("Recall:      ", recall)
        print("F1 Score:    ", f1)

        # convert to dataframe
        df_metrics = pd.DataFrame(
            [
                classifiers,
                samplings,
                features_reductions,
                preds,
                scores,
                accuracies,
                precisions,
                recalls,
                f1s,
                fprs,
                tprs,
                thresholds_rocs,
                roc_aucs,
                prec_prs,
                recall_prs,
                thresholds_prs,
                conf_mat,
                custom_scores,
                durations,
            ],
            index=[
                "classifiers",
                "samplings",
                "features_reductions",
                "preds",
                "scores",
                "accuracies",
                "precisions",
                "recalls",
                "f1s",
                "fprs",
                "tprs",
                "thresholds_rocs",
                "roc_aucs",
                "prec_prs",
                "recall_prs",
                "thresholds_prs",
                "conf_mat",
                "custom_scores",
                "durations",
            ],
        ).T

        # save in pickle format to avoid array being converted to string
        with open("results/metrics.pkl", "wb") as f:
            pickle.dump(df_metrics, f)


if __name__ == "__main__":
    total_start = time.time()
    main()
    print("Total duration is ", time.time() - total_start)
