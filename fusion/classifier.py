# from __future__ import annotations
# from pathlib import Path
# import joblib
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier

# class ScreenClassifier:
#     def __init__(self):
#         self.model = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2, random_state=42,)

#     def train(self, features: np.ndarray, labels: np.ndarray,) -> None:
#         self.model.fit(features, labels,)

#     def predict(self, features: np.ndarray,) -> float:
#         probability = self.model.predict_proba(features.reshape(1, -1))[0][1]
#         return float(probability)

#     def save(self, path: str | Path,) -> None:
#         path = Path(path)
#         path.parent.mkdir(parents=True, exist_ok=True,)
#         joblib.dump(self.model, path,)

#     def load(self, path: str | Path,) -> None:
#         self.model = joblib.load(path)

from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

class ScreenClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42,)

    def train(self, features: np.ndarray, labels: np.ndarray,) -> None:
        paramGrid = {
            "n_estimators": [100, 300, 500],
            "max_depth": [4, 8, 12, None],
            "min_samples_leaf": [1, 2, 4],
            "class_weight": [None, "balanced"],
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42,)

        search = GridSearchCV(
            RandomForestClassifier(random_state=42,),
            paramGrid,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
        )
        search.fit(features, labels,)

        self.model = search.best_estimator_

    def predict(self, features: np.ndarray,) -> float:
        probability = self.model.predict_proba(features.reshape(1, -1))[0][1]
        return float(probability)

    def save(self, path: str | Path,) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True,)
        joblib.dump(self.model, path,)

    def load(self, path: str | Path,) -> None:
        self.model = joblib.load(path)