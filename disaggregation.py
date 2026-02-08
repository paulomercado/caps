import os
import pandas as pd
import numpy as np
from tempdisagg import TempDisaggModel

def prepare_subset(data, date_col, start="1991-01-01", end="1992-12-31"):
    """
    Subset the data between start and end dates
    """
    data[date_col] = pd.to_datetime(data[date_col], dayfirst=True)
    subset = data[(data[date_col] >= start) & (data[date_col] <= end)].copy()
    return subset

def run_tempdisagg(df, method="litterman-opt", conversion="sum"):
    """
    Simple wrapper for TempDisaggModel
    Assumes TempDisaggModel has .fit(df) and .adjust_output(full=False)
    """
    model = TempDisaggModel(method=method, conversion=conversion)
    model.fit(df)
    y_adj = model.adjust_output(full=False)
    return y_adj


def create_disagg_df(data, target_col, x_col=None, yearly=False, grain=3):

    df = pd.DataFrame({'Index': data['Year']})

    if yearly:
        df['Index'] = df['Index']  # yearly
        df['Grain'] = (list(range(1, grain+1)) * len(df))[:len(df)]
    else:
        df['Index'] = df['Index'].astype(int) * 10 + ((df.index // 3) % 4 + 1)
        df['Grain'] = (list(range(1, 4)) * len(df))[:len(df)]

    df['y'] = pd.to_numeric(data[target_col], errors='coerce')

    if x_col is not None:
        df['X'] = pd.to_numeric(data[x_col], errors='coerce')
    else:
        df['X'] = 1.0

    return df

def rolling_tempdisagg(data, method="litterman-opt", conversion="sum", grain=3,start_months=12):
 
    df = data.copy()

    df.iloc[:start_months, df.columns.get_loc('y')] = df.iloc[:start_months]['y'].bfill()

    out = []

    first = df.iloc[:start_months].copy()
    y_adj_first = run_tempdisagg(first, method, conversion)

    out.extend([v[0] if isinstance(v, (list, np.ndarray)) else v for v in y_adj_first])

    # apply temporal disaggregation on rolling window
    for i in range(start_months, len(df)):
        df_roll = df.iloc[:i+1].copy()
   
        if (i+1) % grain == 0: 
            df.iloc[i-grain+1:i+1, df.columns.get_loc('y')] = df.iloc[i-grain+1:i+1]['y'].bfill()
            df_roll = df.iloc[:i+1].copy()

        
        y_adj = run_tempdisagg(df_roll, method, conversion)
        val = y_adj[-1]
        if isinstance(val, (list, np.ndarray)):
            val = val[0]
        out.append(val)  # just the last value
        
    return pd.Series(out)

def main():
    output_file = 'Data/disaggregated.csv'

    # check if already exists
    if os.path.exists(output_file):
        print("Disaggregated CSV exists. Loading...")
        subset = pd.read_csv(output_file)
        return subset

    # load data
    data = pd.read_csv('Data/Macro.csv')

    # create subset
    subset = prepare_subset(data, date_col=data.columns[3], start="1991-01-01", end="2024-12-31")


    #GDP
    df_gdp = create_disagg_df(subset, target_col='Nominal GDP', yearly=False, grain=3)
    df_gdp['X'] = (
        pd.to_numeric(subset['Exports'], errors='coerce') +
        pd.to_numeric(subset['Imports'], errors='coerce')
    ) * pd.to_numeric(subset['USDPHP'], errors='coerce')

    subset['TotalTrade'] =  (pd.to_numeric(subset['Exports'], errors='coerce') +
        pd.to_numeric(subset['Imports'], errors='coerce'))

    subset['TotalTrade_PHPMN'] =  (
        pd.to_numeric(subset['Exports'], errors='coerce') +
        pd.to_numeric(subset['Imports'], errors='coerce')
    ) * pd.to_numeric(subset['USDPHP'], errors='coerce')
    
    subset['NominalGDP_disagg'] = rolling_tempdisagg(df_gdp).values

    #Population
    pop_subset = prepare_subset(data, date_col=data.columns[3], start="1989-01-01", end="2024-12-31")

    dfpop = create_disagg_df(pop_subset, target_col='Population', yearly=True, grain=12)

    pop_roll = rolling_tempdisagg(dfpop, method='denton', conversion='average',grain=12,start_months=36).values
    pop_roll_aligned = pop_roll[24:24 + len(subset)]

    subset['Pop_disagg'] = pop_roll_aligned

    # save
    subset = subset.iloc[:, 3:] 
    subset.to_csv(output_file, index=False)
    print("Disaggregated CSV created.")
    return subset

    
if __name__ == "__main__":
    main()
