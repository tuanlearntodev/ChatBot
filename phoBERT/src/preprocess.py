import pandas as pd


def load_books(file_path):
    df = pd.read_json(file_path)
    return df
