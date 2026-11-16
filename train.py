import pandas as pd
import mlflow
import logging
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import mlflow.sklearn
import os

logging.basicConfig(level= logging.INFO, format= "%(asctime)s - %(levelname)s - %(message)s")
logger= logging.getLogger(__name__)

try:
    logging.info('loading dataset')
    df= pd.read_csv('../data/cleaned_dataset/cleaned.csv')
    logging.info('Dataset loaded successfully')
    X= df.drop('Medical Insurance Expenses', axis= 1)
    y= df['Medical Insurance Expenses']
    X_train, X_test, y_train, y_test= train_test_split(X,y, test_size= 0.2, random_state= 42)
    num_cols= X.select_dtypes(include= ['int', 'float']).columns
    cat_cols= X.select_dtypes(include= ['object']).columns
    num_pipeline= Pipeline([
        ('impute', SimpleImputer(strategy= 'median'))
        ('scaling', StandardScaler())
    ])
    cat_pipeline= Pipeline([
        ('impute', SimpleImputer(strategy= 'most_frequent')),
        ('encoding', OneHotEncoder())
    ])
    preprocessing= ColumnTransformer([
        ('num', num_pipeline, num_cols)
        ('cat', cat_pipeline, cat_cols)
    ])
    models= {
        'Decision_Tree': DecisionTreeRegressor(),
        'Random_Forest': RandomForestRegressor(),
        'Gradient_Boosting': GradientBoostingRegressor(),
        'XG_Boost': XGBRegressor()
    }

    mlflow.set_tracking_uri('https://localhost:5000')
    mlflow.set_experiment('bank_insurence')

    best_model= None
    best_model_name= None
    best_rmse= float('inf')

    for model_name, model in models.item():
        pipeline= Pipeline([
            ('preprocessing', preprocessing),
            ('model', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred= pipeline.predict(X_test)
        logger.info('Training successfull')

        with mlflow.start_run(run_name= 'bank_insurence'):
            mae= mean_absolute_error(y_test, y_pred)
            mse= mean_squared_error(y_test, y_pred)
            rmse= root_mean_squared_error(y_test, y_pred)
            r2= r2_score(y_test, y_pred)

            logger.info('Evaluation metrics calcu;ated successfully')
            mlflow.log_params('model_name', model_name)
            mlflow.log_metric('mae', mae)
            mlflow.log_metric('mse', mse)
            mlflow.log_metric('rmse', rmse)
            mlflow.sklearn.log_model(model, serialization_format= 'cloudpickle', artifact_path= 'model')

            if rmse< best_rmse:
                best_rmse= rmse
                best_model= pipeline
                best_model_name= model_name

    os.mkdir('models', exist_ok= True)
    joblib.dump(best_model, 'models'/'best_model.pkl')
    logger.info('best_model: %s', best_model_name)
    logger.info('best_rmse: %s', best_rmse)
    logger.info('model saved successfully')

except Exception:
    logger.exception('Model Training Failed')    