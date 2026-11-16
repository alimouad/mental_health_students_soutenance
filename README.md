# MindScope

MindScope is an educational machine-learning project for predicting student
mental-health risk levels from academic, lifestyle, and support-related
features. It includes a Streamlit dashboard, an Apache Airflow pipeline, and
notebooks for exploration, preprocessing, training, and hyperparameter
optimization.

> **Important:** This project is for educational purposes only. Its
> predictions are not medical diagnoses or a clinical decision-making tool.

## Project structure

```text
.
├── app/                         # Streamlit application
├── dags/                        # Airflow DAG
├── data/
│   ├── raw/                     # Original dataset
│   └── processed/               # Train/test data and model comparison
├── models/                      # Saved model artifacts
├── notebooks/                   # Analysis and machine-learning notebooks
├── reports/                     # Evaluation reports
├── src/
│   ├── evaluation/              # Evaluation helpers
│   ├── models/                  # Training and model selection
│   └── preprocessing/           # Feature preprocessing
├── docker-compose.yml
├── requirements.txt
└── requirements-airflow.txt
```

## Models

The project compares:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest

The current selected model is **Logistic Regression**. The latest recorded
cross-validation F1-macro score is approximately **0.7633**.

The latest evaluation report contains:

| Metric | Score |
|---|---:|
| Accuracy | 72.75% |
| Precision | 73.14% |
| Recall | 72.63% |
| F1-score | 72.84% |

These values are stored in [`reports/metrics.json`](reports/metrics.json).

## Run with Docker Compose

Make sure Docker and Docker Compose are installed, then run:

```bash
docker compose up -d --build
docker compose ps
```

The services are available at:

- **Streamlit:** http://localhost:8502
- **Airflow:** http://localhost:8081

To follow logs:

```bash
docker compose logs -f streamlit
docker compose logs -f airflow
```

To stop the services:

```bash
docker compose down
```

The Airflow container uses the `airflow` user. The Compose configuration grants
it access to the `models` and `reports` directories so training outputs can be
saved to the host filesystem.

### Airflow login

Airflow runs in standalone mode and generates its admin password during
initialization. Retrieve the credentials with:

```bash
docker compose logs airflow | grep -i password
```

Use the displayed username and password at http://localhost:8081.

## Run Streamlit locally

Create or activate a Python environment, install the dependencies, and start
the application:

```bash
python -m pip install -r requirements.txt
streamlit run app/app.py
```

The application loads the raw dataset from
`data/raw/student_mental_health.csv` and the saved model from
`models/best_mental_health_model.pkl`.

The dashboard provides:

1. Dataset overview and target distribution
2. Model performance metrics and confusion matrix
3. CSV upload and student-risk predictions
4. Project information and limitations

Uploaded prediction files must contain all feature columns expected by the
model. The `Student_ID` and `Mental_Health_Risk` columns are not required for
prediction input.

## Run the Airflow pipeline

The DAG is named `mindscope_pipeline` and contains three tasks:

```text
prepare_data → train_model → evaluate_model
```

Trigger it from the Airflow interface, or inspect it with:

```bash
docker exec mindscope-airflow airflow dags list
docker exec mindscope-airflow airflow dags trigger mindscope_pipeline
```

The pipeline reads processed train/test files from `data/processed/`, trains
the configured models using cross-validation, saves the best pipeline to
`models/mindscope_model.joblib`, and writes reports to `reports/`.

## Notebooks

The notebooks are intended to be run from the `notebooks/` directory:

- `data_exploratory.ipynb`: exploratory data analysis
- `Data_Preprocessing.ipynb`: preprocessing and dataset preparation
- `Model_Training.ipynb`: model training and comparison
- `Hyperparameter_Optimization.ipynb`: GridSearchCV optimization

## Development checks

Validate the Compose file before starting the services:

```bash
docker compose config --quiet
```

Check service status:

```bash
docker compose ps
```

## Data and privacy

Do not commit credentials, private datasets, or generated secrets. Keep
environment-specific values in `.env` and review data files before sharing the
repository.