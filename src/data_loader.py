## Load, Pre-Process, Feature Engineering, Encode Data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

def load_data(file_path):
    df = pd.read_csv(file_path, header=0, encoding='unicode_escape')
    return df

def preprocess_data(df):
    # Converting date columns to datetime format
    df['order date'] = pd.to_datetime(df['order date (DateOrders)'])
    df['shipping date'] = pd.to_datetime(df['shipping date (DateOrders)'])

    # Extracting year, month, day, hour, minute from 'order date' and 'shipping date'
    df['order year'] = df['order date'].dt.year
    df['order month'] = df['order date'].dt.month
    df['order day'] = df['order date'].dt.day
    df['order hour'] = df['order date'].dt.hour
    df['order minute'] = df['order date'].dt.minute

    df['shipping year'] = df['shipping date'].dt.year
    df['shipping month'] = df['shipping date'].dt.month
    df['shipping day'] = df['shipping date'].dt.day
    df['shipping hour'] = df['shipping date'].dt.hour
    df['shipping minute'] = df['shipping date'].dt.minute

    return df

def select_features(df):
    # Selecting important features
    features = df.loc[:, ['Type', 'Delivery Status', 'Sales per customer', 'Days for shipping (real)',
                          'Days for shipment (scheduled)', 'order year', 'order month', 'order day',
                          'order hour', 'order minute', 'Benefit per order', 'Category Name', 'Latitude',
                          'Longitude', 'Customer Segment', 'Customer City', 'Customer Country', 'Customer State',
                          'Department Name', 'Market', 'Order City', 'Order Country', 'Order Item Discount',
                          'Order Item Product Price', 'Order Item Quantity', 'Order Item Total', 'Order State',
                          'Order Region', 'Product Name', 'shipping year', 'shipping month', 'shipping day',
                          'shipping hour', 'shipping minute', 'Shipping Mode', 'Late_delivery_risk', 'Order Status',
                          'Sales', 'Product Price', 'Order Profit Per Order', 'Order Item Discount Rate', 
                          'Order Item Profit Ratio']]
    
    return features

def encode_data(data):
    # Encoding categorical variables

    # Mapping 'Order Status' to binary
    data['Order Status'] = [0 if status != 'SUSPECTED_FRAUD' else 1 for status in data['Order Status']]

    enc = LabelEncoder()
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = enc.fit_transform(data[col])
    
    return data

def split_data(data):
    # Splitting features and target
    y = data['Order Status']
    X = data.drop(['Order Status'], axis=1)
    
    # Splitting into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
    
def scaled_data(x_train, x_test):    
    # Feature scaling
    scaler = StandardScaler().fit(x_train)
    X_train_scaled = scaler.transform(x_train)
    X_test_scaled = scaler.transform(x_test)

    return X_train_scaled, X_test_scaled

def apply_smote(X_train, y_train):
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    return X_train_res, y_train_res
