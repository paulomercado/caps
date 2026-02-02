from tempdisagg import TempDisaggModel
import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as ts 

def process_gdp(data_path):
        """
        Loads and processes Macro.csv to create Index, Grain, X, y
        Returns a dataframe ready for TempDisagg
        """
        data = pd.read_csv(data_path)
        
        df = pd.DataFrame({'Index': data['Year']})
        df['Index'] = df['Index'].astype(int) * 10 + ((df.index // 3) % 4 + 1)
        df['Grain'] = (list(range(1, 4)) * len(df))[:len(df)]
        df['y'] = data['Nominal GDP']
        df['y'] = df.groupby('Index')['y'].bfill()
        
        df['X'] = (
            pd.to_numeric(data['Exports'], errors='coerce') +
            pd.to_numeric(data['Imports'], errors='coerce')
        ) * pd.to_numeric(data['USDPHP'], errors='coerce')
        
        df = df[(df['Index'] // 10).between(1991, 2024)].reset_index(drop=True)
        
        return df

def load_data():
     return False

class Disagg:
    def __init__(self, method="litterman-opt", conversion="sum"):
        self.method = method
        self.conversion = conversion
        self.model = None
    
    def fit_predict(self, Index, Grain, X, y):
        """
        Inputs:
            Index: pandas Series or list
            Grain: pandas Series or list
            X: pandas Series or list of regressors
            y: pandas Series or list of target
        Returns:
            df_out: pandas DataFrame with Index, Grain, X, y, predicted
        """
        df = pd.DataFrame({
            "Index": Index,
            "Grain": Grain,
            "X": X,
            "y": y
        })
        
        # Optional: fill missing y values within Index groups
        df['y'] = df.groupby('Index')['y'].bfill()

        # Fit model
        self.model = TempDisaggModel(method=self.method, conversion=self.conversion)
        self.model.fit(df)
        
        # Predict
        y_hat = self.model.predict(full=False)
        df['predicted'] = y_hat
        
        return df