import boto3
from io import BytesIO
from pathlib import Path
import pandas as pd
from urllib.parse import urlparse
import logging

logging.basicConfig(level= logging.INFO, format= "%(asctime)s - %(levelname)s - %(message)s")
logger= logging.getLogger(__name__)

def load_data(source, source_type, query= None):
    s3_path= urlparse(source)  
    key= s3_path.path.lstrip('/')
    bucket= s3_path.netloc

    s3= boto3.client("s3")
    response= s3.get_object(Bucket= bucket, Key= key)
    file_data= BytesIO(response["Body"].read())
    file_type= Path(key).suffix.lower()
    try:
        if file_type == ".csv":
            df= pd.read_csv(file_data)
        elif file_type in ['xlsx', 'xls']:
            df= pd.read_excel(file_data)
        elif file_type == ".json":
            df= pd.read_json(file_data)
        else:
            raise ValueError('Unsupported File type')
        if df.empty:
            raise ValueError('the dataset is empty')
        logger.info('Data Ingestion succeded')
        return df
    except Exception:
             logger.exception('Data Ingestion Failed')           


if __name__ == "__main__":
     df= load_data("s3://insurence-bucket-ssd023/insurance.csv", source_type= "s3")
     output_path= Path("data/ingested/bank_data_injested.csv")
     output_path.parent.mkdir(parents= True, exist_ok= True)

     df.to_csv(output_path, index= False)
     logger.info('ingested data saved successfully')