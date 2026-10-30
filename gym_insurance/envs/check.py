# Install necessary libraries if not already installed
#!pip install pandas requests openpyxl pyarrow

# Import libraries
import os
import requests
import pandas as pd
import logging

# Setup logging configuration
logs_dir = '/content/logs'
os.makedirs(logs_dir, exist_ok=True)
log_file = os.path.join(logs_dir, 'psr_preprocessing.logs')

# Configure logging
logging.basicConfig(
    filename=log_file,
    filemode='a',  # append mode
    level=logging.DEBUG,  # Capture both INFO and DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Log the beginning of the process
logging.info("Starting PSR dataset processing.")

# Directory to store the datasets
save_dir = '/content/sisser_data'
os.makedirs(save_dir, exist_ok=True)
logging.info(f"Created save directory: {save_dir}")

# URLs for the Excel datasets
datasets = {
    "PSR_2006_2015": "https://dados.agricultura.gov.br/dataset/baefdc68-9bad-4204-83e8-f2888b79ab48/resource/dc01ccf6-d328-41a6-842d-7eff9b08f6d6/download/psrdadosabertos2006a2015excel.xlsx",
    "PSR_2016_2021": "https://dados.agricultura.gov.br/dataset/baefdc68-9bad-4204-83e8-f2888b79ab48/resource/2c5c55d0-1473-4749-b08f-cfaf887a9fa3/download/dicionariodedados-sisser.pdf",
    "PSR_2022_2023": "https://dados.agricultura.gov.br/dataset/baefdc68-9bad-4204-83e8-f2888b79ab48/resource/4f90f7b7-82f7-4ff3-bb25-abb660e3ae77/download/psrdadosabertos2022a2023excel.xlsx",
    "PSR_2024": "https://dados.agricultura.gov.br/dataset/baefdc68-9bad-4204-83e8-f2888b79ab48/resource/b904117d-b758-406d-92ef-4c7762017c61/download/psrdadosabertos2024excel.xlsx"
}

# Download and load datasets directly with pandas.read_excel
dataframes = {}
for name, url in datasets.items():
    logging.info(f"Downloading and loading dataset: {name} from {url}")
    df = pd.read_excel(url)
    df.columns = [col.lower() for col in df.columns]  # Convert column names to lowercase
    dataframes[name] = df
    logging.info(f"Loaded dataset {name} with shape {df.shape}")
    logging.debug(f"First 5 rows of {name}:\n{df.head()}")

# Find common columns across all datasets
common_columns = set.intersection(*[set(df.columns) for df in dataframes.values()])
logging.info(f"Common columns across all datasets: {common_columns}")

# Concatenate datasets keeping only the common columns
concatenated_df = pd.concat([df[list(common_columns)] for df in dataframes.values()], ignore_index=True)
logging.info(f"Concatenated dataset shape: {concatenated_df.shape}")

# Persist the final DataFrame as a Parquet file
final_parquet_path = os.path.join(save_dir, "final_sisser_data.parquet")
concatenated_df.to_parquet(final_parquet_path, index=False)
logging.info(f"Final concatenated data saved to {final_parquet_path}")

# Save processed datasets separately to CSV files (optional)
for name, df in dataframes.items():
    csv_path = os.path.join(save_dir, f"{name}.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved {name} as CSV to {csv_path}")

# Optionally, save the final Parquet file to Google Drive
from google.colab import drive
drive.mount('/content/drive')

drive_parquet_path = '/content/drive/My Drive/final_sisser_data.parquet'
concatenated_df.to_parquet(drive_parquet_path, index=False)
logging.info(f"Final concatenated data saved to Google Drive at {drive_parquet_path}")

# Log the completion of the process
logging.info("PSR dataset processing completed.")