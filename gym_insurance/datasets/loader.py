import pandas as pd
import importlib.resources as pkg_resources

import json

with open('config.json', 'r') as config_file:
    config = json.load(config_file)


def load_psr_data(dataset="PSR - 2016 a 2021"):
    data_files =config['datasets']
    if dataset not in data_files:
        raise ValueError(f"Dataset {dataset} not found.")

    file_name = data_files[dataset]
    with pkg_resources.open_text('gym_insurance.datasets', file_name,  encoding='latin1') as file:
        df = pd.read_csv(file, encoding='latin-1', sep = ';', skiprows=[526])
        #df.str.replace(",", ".", inplace=True)
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.replace(",", ".")
        df.columns = df.columns.str.lower()
        df["valor_indenização"] = df["valor_indenização"].astype(float)
        print(df.columns)
    return df.loc[:, ["id_proposta", "dt_proposta", "nr_documento_segurado", "valor_indenização"]].query("valor_indenização>0")