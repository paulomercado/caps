from tempdisagg import TempDisaggModel
import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as ts 


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