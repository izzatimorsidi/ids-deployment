import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

scaler = joblib.load('models/standard_scaler.pkl')

def preprocess_data(file_path):
    df = pd.read_csv(file_path, encoding='latin1')

    df = df.replace([np.inf, -np.inf], 100000)
    
    df['Time'] = pd.to_datetime(df['Time'])  

    df['Length'] = df['Length'].astype(float)

    df['Flow'] = df['Source'].astype(str) + '-' + df['Destination'].astype(str)

    df['Total Fwd Packets'] = df.groupby('Flow')['No.'].transform('count')    
    df['Max Packet Length'] = df.groupby('Flow')['Length'].transform('max')   
    df['Flow Start'] = df.groupby('Flow')['Time'].transform('min')
    df['Flow End'] = df.groupby('Flow')['Time'].transform('max')
    df['Flow Duration'] = (df['Flow End'] - df['Flow Start']).dt.total_seconds() 
       
    df['Total Flow Bytes'] = df.groupby('Flow')['Length'].transform('sum')
    df['Average Packet Size'] = df['Total Flow Bytes'] / df['Total Fwd Packets']
        
    df['Flow Bytes/s'] = df['Total Flow Bytes'] / df['Flow Duration']
    df['Flow Packets/s'] = df['Total Fwd Packets'] / df['Flow Duration']

    
    df['Flow Bytes/s'] = df.apply(lambda x: x['Total Flow Bytes'] / x['Flow Duration'] if x['Flow Duration'] > 0 else 0, axis=1)    
    df['Flow Packets/s'] = df.apply(lambda x: x['Total Fwd Packets'] / x['Flow Duration'] if x['Flow Duration'] > 0 else 0, axis=1)   

    
    features_for_scaling = df[['Total Fwd Packets', 'Max Packet Length', 'Flow Duration', 
                               'Average Packet Size', 'Flow Bytes/s', 'Flow Packets/s']]
    
    features_for_scaling.replace([np.inf, -np.inf], np.nan, inplace=True)  
    features_for_scaling.fillna(0, inplace=True)

    scaled_features = scaler.transform(features_for_scaling)
  
    processed_features = df[['No.', 'Time', 'Source', 'Destination', 'Protocol', 'Length']]

    return scaled_features, processed_features, df
