#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import os, argparse, numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

CHUNK_SIZE = 25000          
USECOLS  = ['tpep_pickup_datetime', 'tpep_dropoff_datetime',
            'passenger_count', 'trip_distance', 'RatecodeID',
            'PULocationID', 'DOLocationID', 'payment_type', 'extra',
            'total_amount']
DTYPES = {                     
    'passenger_count': 'float64',
    'RatecodeID'     : 'float64',
    'payment_type'   : 'float64',
    'trip_distance'  : 'float64',
    'extra'          : 'float32',
    'total_amount'   : 'float64'
}

def iter_clean_csv(path):
    """Read and clean CSV in chunks."""
    dfs = []
    for k, chunk in enumerate(pd.read_csv(path, usecols=USECOLS,
                                          dtype=DTYPES,
                                          date_format='%m/%d/%Y %I:%M:%S %p',
                                          parse_dates=['tpep_pickup_datetime',
                                                       'tpep_dropoff_datetime'],
                                          chunksize=CHUNK_SIZE)):
        if k % 10 == 0:
            print(f'[INFO] Processing chunk {k}, size {len(chunk)}')
        # ---- basic features ----
        chunk = chunk.dropna()
        chunk = chunk[(chunk.total_amount > 0) & (chunk.trip_distance > 0)]
        # ---- time features ----
        # duration
        chunk['trip_duration'] = (
            chunk['tpep_dropoff_datetime'] - chunk['tpep_pickup_datetime']
        ).dt.total_seconds() / 60.0
        chunk = chunk[chunk['trip_duration'].between(0.5, 360)]
        # time features
        dt = chunk['tpep_pickup_datetime'].dt
        chunk['pickup_weekday']  = dt.weekday.astype('int8')
        chunk['pickup_time']     = dt.hour.astype('int8') + dt.minute.astype('int8')/60.0       # [0-24)
        # int8
        chunk['passenger_count'] = chunk['passenger_count'].astype('int8')
        chunk['RatecodeID'] = chunk['RatecodeID'].astype('int8')
        chunk['payment_type'] = chunk['payment_type'].astype('int8')

        '''
        for col in ['passenger_count','RatecodeID','PULocationID',
                    'DOLocationID','payment_type','extra']:
            chunk[col] = chunk[col].astype(str)
            chunk[col] = LabelEncoder().fit_transform(chunk[col])        
        '''
        # ---- final columns ----
        out_cols = ['passenger_count', 'RatecodeID', 'PULocationID',
                    'DOLocationID', 'payment_type', 'extra', 'trip_distance',
                    'trip_duration', 'pickup_weekday', 'pickup_time',
                    'total_amount']
        out_cols = ['passenger_count', 'RatecodeID', 'PULocationID',
                    'DOLocationID', 'payment_type', 'extra', 'trip_distance',
                    'trip_duration', 'total_amount']
        dfs.append(chunk[out_cols])
    return pd.concat(dfs, ignore_index=True)

def build_array(df):
    """DataFrame -> X, y"""
    feat = ['passenger_count','RatecodeID','PULocationID','DOLocationID',
            'payment_type','extra','trip_distance',
            'trip_duration', 'pickup_weekday', 'pickup_time'
            ]
    feat = ['passenger_count','RatecodeID','PULocationID','DOLocationID',
            'payment_type','extra','trip_distance',
            'trip_duration'
            ]
    X = df[feat].values 
    y = df['total_amount'].values.reshape(-1,1)
    return X, y

def save_splits(X, y, out_dir, test_size=0.3, random_state=42):
    os.makedirs(out_dir, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    xsc, ysc = MinMaxScaler(), MinMaxScaler()
    X_train = xsc.fit_transform(X_train)
    X_test  = xsc.transform(X_test)
    y_train = ysc.fit_transform(y_train)
    y_test  = ysc.transform(y_test)
    np.savez(f'{out_dir}/train_all.npz', X=X_train, y=y_train)
    np.savez(f'{out_dir}/test_all.npz',  X=X_test,  y=y_test)
    np.savez(f'{out_dir}/scaler_all.npz',
             x_scale=xsc.scale_, x_min=xsc.data_min_,
             y_scale=ysc.scale_[0], y_min=ysc.data_min_[0])
    print(f'[INFO] Train set {X_train.shape} Test set {X_test.shape}')

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', dest='csv', default='data/ny5m.csv')
    parser.add_argument('--out_dir', default='data/')
    args = parser.parse_args()
    start_time = time.time()
    print('[INFO] Start chunked cleaning …')
    df = iter_clean_csv(args.csv)
    print(df.head(3))
    print(df.tail(3))
    print(f'[INFO] Size of the cleaned dataset {df.shape}')
    print(f'[INFO] Cleaning time {time.time()-start_time:.2f} s')
    X, y = build_array(df)
    save_splits(X, y, args.out_dir)

if __name__ == '__main__':
    main()
