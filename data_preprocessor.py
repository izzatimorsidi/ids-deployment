import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

scaler = joblib.load('models/standard_scaler.pkl')

def preprocess_data(file_path):
    df = pd.read_csv(file_path, encoding='latin1')

    # Replace 'inf' and '-inf' values with 100000 for all columns
    df = df.replace([np.inf, -np.inf], 100000)
    
    # Convert 'Time' to a more usable format if necessary
    df['Time'] = pd.to_datetime(df['Time'])
    
    df['Length'] = df['Length'].astype(float)

    # Identify flows based on Source and Destination
    df['Flow'] = df['Source'].astype(str) + '-' + df['Destination'].astype(str)
    
    # Calculate additional features for each flow
    # Total Forward Packets
    df['Total Fwd Packets'] = df.groupby('Flow')['No.'].transform('count')
    
    # Max Packet Length
    df['Max Packet Length'] = df.groupby('Flow')['Length'].transform('max')
    
    # Flow Duration
    df['Flow Start'] = df.groupby('Flow')['Time'].transform('min')
    df['Flow End'] = df.groupby('Flow')['Time'].transform('max')
    df['Flow Duration'] = (df['Flow End'] - df['Flow Start']).dt.total_seconds()
    
    # Average Packet Size
    df['Total Flow Bytes'] = df.groupby('Flow')['Length'].transform('sum')
    df['Average Packet Size'] = df['Total Flow Bytes'] / df['Total Fwd Packets']
    
    # Flow Bytes/s and Flow Packets/s
    df['Flow Bytes/s'] = df['Total Flow Bytes'] / df['Flow Duration']
    df['Flow Packets/s'] = df['Total Fwd Packets'] / df['Flow Duration']

    # Example for Flow Bytes/s, handling division by zero
    df['Flow Bytes/s'] = df.apply(lambda x: x['Total Flow Bytes'] / x['Flow Duration'] if x['Flow Duration'] > 0 else 0, axis=1)

# Example for Flow Packets/s
    df['Flow Packets/s'] = df.apply(lambda x: x['Total Fwd Packets'] / x['Flow Duration'] if x['Flow Duration'] > 0 else 0, axis=1)

# For any ratio or division, ensure the denominator is not zero; otherwise, assign a default value

    

    # Select and scale the features
    features_for_scaling = df[['Total Fwd Packets', 'Max Packet Length', 'Flow Duration', 
                               'Average Packet Size', 'Flow Bytes/s', 'Flow Packets/s']]
    
# Before scaling, replace infinite values and handle division by zero
    features_for_scaling.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf with NaN

# Fill NA/NaN values which might result from division by zero or replacing infinities
# You might choose a value or a strategy that makes sense for your data and model
    features_for_scaling.fillna(0, inplace=True)

# Now, you should be able to scale without encountering the ValueError
    scaled_features = scaler.transform(features_for_scaling)


    # Features for display in prediction results
    processed_features = df[['No.', 'Time', 'Source', 'Destination', 'Protocol', 'Length', 'Info']]

    return scaled_features, processed_features, df
