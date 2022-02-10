from typing import Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from constants import DATA, TEST_RATIO


def prepare_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns data in the form of training and testing data."""
    df = pd.read_csv(DATA / "HTRU_2.csv")
    X = np.array(df.drop(["Classifier"], 1), dtype=np.float32)
    y = np.array(df["Classifier"], dtype=np.float32)
    sc = StandardScaler()

    X_resampled, y_resampled = SMOTE(random_state=43).fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=TEST_RATIO
    )

    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.fit_transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test


def test_SVC(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> None:
    """Shows accuracy and classification w.r.t SVC"""
    svm_rbf = SVC(kernel="rbf")
    svm_rbf.fit(X_train, y_train)
    svm_rbf_pred = svm_rbf.predict(X_test)

    print(
        "Accuracy Score of Super Vector Machine, RBF kernel: ",
        accuracy_score(y_test, svm_rbf_pred),
    )

    print(classification_report(y_test, svm_rbf_pred))


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = prepare_data()
    test_SVC(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
