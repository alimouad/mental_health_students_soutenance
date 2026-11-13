from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC

from src.preprocessing.preprocessor import create_preprocessor

def train_models(X_train , y_train):

    preprocessor = create_preprocessor(X_train)

    models = {


        "Logistic Regression": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]),

        "SVM": Pipeline(steps=[
            ("preprocessor" ,preprocessor),
            ("classifier", SVC(random_state=42))
        ]),

        "Random Forest": Pipeline(steps=[
            ("preprocessor" ,preprocessor),
            ("classifier" ,RandomForestClassifier(random_state=42))
        ])

    }


    params_grids = {
        "Logistic Regression": {
            "classifier__C": [0.1, 1 ,10]
        },

        "SVM": {
            "classifier__C": [0.1, 1, 10],
            "classifier__kernel": ["linear", "rbf"],
            "classifier__class_weight": [None, "balanced"]
        },

        "Random Forest": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_split": [2, 5, 10]
        }
    }


    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    grid_results = {}


    for name , pipeline in models.items():
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=params_grids[name],
            cv=cv,
            scoring="f1_macro",
            n_jobs=1
        )

        grid_search.fit(X_train, y_train)
        grid_results[name] = grid_search


        best_name = max(
            grid_results,
            key=lambda name: grid_results[name].best_score_
        )

        best_search = grid_results[best_name]

    return grid_results, best_name, best_search.best_estimator_