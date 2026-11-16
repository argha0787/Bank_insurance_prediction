
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib

# Prometheus imports
from prometheus_client import Counter, Histogram, make_asgi_app
import time


# 1. FastAPI application
app = FastAPI('Medical Insurance API')


# 2. Load trained model
model = joblib.load(
    Path(__file__).resolve().parents[1] / 'models' / 'best_model.pkl'
)


# 3. Prometheus metrics

# Count total API requests
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total number of API requests'
)

# Count total predictions
PREDICTION_COUNT = Counter(
    'predictions_total',
    'Total number of predictions'
)

# Count failed predictions
ERROR_COUNT = Counter(
    'prediction_errors_total',
    'Total number of prediction errors'
)

# Measure prediction response time
PREDICTION_TIME = Histogram(
    'prediction_duration_seconds',
    'Time taken to generate predictions'
)


# 4. Input schema
class Input_Data(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str


# 5. Home endpoint
@app.get('/')
def home():

    REQUEST_COUNT.inc()

    return {'message': 'The app is running successfully'}


# 6. Prediction endpoint
@app.post('/predict')
def predict_insurense_expense(input_data: Input_Data):

    REQUEST_COUNT.inc()

    try:
        start_time = time.time()

        data = pd.DataFrame([input_data.model_dump()])

        predict = model.predict(data)

        PREDICTION_TIME.observe(time.time() - start_time)

        PREDICTION_COUNT.inc()

        return {
            'message': float(predict[0])
        }

    except Exception:

        ERROR_COUNT.inc()

        raise


# 7. Prometheus metrics endpoint
metrics_app = make_asgi_app()

app.mount('/metrics', metrics_app)
