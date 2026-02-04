import os
import pandas as pd
import numpy as np
from tempdisagg import TempDisaggModel

def prepare_subset(data, date_col, start="1991-01-01", end="2024-12-31"):
    data[date_col] = pd.to_datetime(data[date_col], dayfirst=True)
    subset = data[(data[date_col] >= start) & (data[date_col] <= end)].copy()
    return subset

def create_disagg_df(data, target_col, x_col=None, yearly=False, grain=3):
    df = pd.DataFrame({'Index': data['Year']})

    if yearly:
        df['Index'] = df['Index']  # yearly
        df['Grain'] = (list(range(1, grain+1)) * len(df))[:len(df)]
    else:
        df['Index'] = df['Index'].astype(int) * 10 + ((df.index // 3) % 4 + 1)
        df['Grain'] = (list(range(1, 4)) * len(df))[:len(df)]


    df['y'] = pd.to_numeric(data[target_col], errors='coerce')

    df['y'] = df.groupby('Index')['y'].bfill()

    if x_col is not None:
        df['X'] = pd.to_numeric(data[x_col], errors='coerce')
    else:
        df['X'] = 1.0

    return df

def run_tempdisagg(df, method="litterman-opt", conversion="sum"):
    model = TempDisaggModel(method=method, conversion=conversion)
    model.fit(df)
    y_adj = model.adjust_output(full=False)
    return y_adj

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
    subset = prepare_subset(data, date_col=data.columns[3])

    # Disaggregate Nominal GDP
    
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
    subset['NominalGDP_disagg'] = run_tempdisagg(df_gdp, method="litterman-opt", conversion="sum")

    # Disaggregate Population (monthly)
    df_pop = create_disagg_df(subset, target_col='Population', yearly=True, grain=12)
    subset['Population_Disagg'] = run_tempdisagg(df_pop, method="denton", conversion="average")

    # save
    subset = subset.iloc[:, 3:] 
    subset.to_csv(output_file, index=False)
    print("Disaggregated CSV created.")
    return subset

    
if __name__ == "__main__":
    main()
