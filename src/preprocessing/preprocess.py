import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import (
    DataLoader, 
    Dataset
    )
from sklearn.model_selection import train_test_split
import sys, os

def create_rolling_ts(
    input_data, 
    lookback=5, 
    return_np=False
    ):
    """
    Make flat raw_data by using pd.concat instead, pd.concat([df1, df2]).
    Slow function.
    Save raw_data as preprocessed?
    """
    x = []
    y = []
    rows = len(input_data)
    features = input_data.copy()
    target = input_data.copy()
    for i in range(rows - lookback):
        rolling_features = features.iloc[i: i + lookback]
        rolling_target = target.iloc[i + lookback: i + lookback + 1]
        x.append(rolling_features)
        y.append(rolling_target)
    if return_np:
        return np.array(x), np.array(y)
    return x, y


def date_features(df):
    # Check if index is datetime.
    if isinstance(df, pd.core.series.Series):
        df = pd.DataFrame(df, index=df.index)

    df.loc[:, 'day_of_year'] = df.index.dayofyear
    df.loc[:, 'month'] = df.index.month
    df.loc[:, 'day_of_week'] = df.index.day
    df.loc[:, 'hour'] = df.index.hour
    return df
    

def ts_split(raw_data):
    train_size = int(len(raw_data) * 0.75)
    train_set = raw_data.iloc[ : train_size]
    valid_set = raw_data.iloc[train_size : ]
    return train_set, valid_set


def is_pandas(raw_data):
    return isinstance(
        raw_data, 
        (pd.core.frame.DataFrame, pd.core.series.Series)
        )


def  add_datepart(raw_data):
    raw_data = date_features(raw_data)
    return raw_data


class ContCatSplit:
    def __init__(
        self, 
        raw_data, 
        add_date=False,
        cat_types=None
        ):

        self.raw_data = raw_data.copy()
        self.add_date = add_date
        self.cat_types = cat_types

    def cont_cat_split(self):
        try:
            if is_pandas(self.raw_data):
                if self.add_date:
                    self.raw_data = add_datepart(self.raw_data)
                self.cat = self.raw_data.select_dtypes(include=self.cat_types)
                cat_cols = self.cat.columns
                self.cont = self.raw_data.drop(cat_cols, axis=1)
                return self.cont, self.cat
        except Exception as e:
            print(f'from cont_cat_split: {e}')


class ToTorch(Dataset):

    def __init__(
            self,
            features,
            target
            ):
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx]
        target = self.target[idx]
        return {
            'features': torch.from_numpy(np.array(features)).float(), 
            'target': torch.from_numpy(np.array(target)).float()
            }


class Scaler:
    def __init__(self, scaler_name='standard'):
        if scaler_name == 'standard':
            self.transform = StandardScaler()
        elif scaler_name == 'minmax':
            self.transform = MinMaxScaler()
        else:
            pass

    def train_transform(self, xtrain):
        xtrain_np = xtrain.copy()
        return self.transform.fit_transform(xtrain_np)

    def test_transform(self, xtest):
        return self.transform.transform(xtest)

    def inverse(self, x):
        return self.transform.inverse_transform(x)


class Normalize:
    def __init__(self, train_ds, valid_ds, test_ds=None):
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds

        train_ds = self.train_ds.copy()
        self.avg = self.set_mean(train_ds)
        self.st_dev = self.set_std(train_ds)

    def fit(self):
        try:
            self.train_ds = self.z_norm(self.train_ds)
            self.valid_ds = self.z_norm(self.valid_ds)
            if self.test_ds is not None:
                self.test_ds = self.z_norm(self.test_ds)
                return self.train_ds, self.valid_ds, self.test_ds
            return self.train_ds, self.valid_ds
        except:
            raise

    def z_norm(self, x):
        return (x[None, :, None] - self.avg)/self.st_dev        

    def set_mean(self, data):
        """Check dimensions"""
        return np.nanmean(data)

    def set_std(self, x):
        """Check dimensions"""
        return np.nanstd(x)

    def inverse_transform(self):
        pass


class DataScaler:
    """This class can be implemented in local folder"""
    def __init__(
        self, 
        raw_data,
        normalize=True, 
        is_sequential=True,
        scaler_name='standard',
        # add_dates=True,
        cat_types=['int64']
        ):
        self.raw_data = raw_data
        self.normalize = normalize
        self.is_sequential = is_sequential
        self.scaler_name = scaler_name
        # self.add_dates = add_dates
        self.cat_types = cat_types
        """
        Categorical and numerical data 
        go through different pipelines.
        1) Numeric data has to be normaziled.
        2) Categorical data does not need Normalization
        """

    def fit(self, train_ds, valid_ds, test_ds=None):
        """
        Use Normalize class.
        Apply:
        0. check how many elem from self.dataset 
        1. fit_transform
        2. transform
        """
        self.feature_scaler = Normalize(train_ds, valid_ds, test_ds)
        # self.target_scaler = Normalize(self.scaler_name)
        if test_ds is not None:
            self.train_ds, self.valid_ds, self.test_ds = self.feature_scaler.fit()
            return self.train_ds, self.valid_ds, self.test_ds
        else:
            self.train_ds, self.valid_ds = self.feature_scaler.fit()
            return self.train_ds, self.valid_ds

    def run_pipeline(self):
        """Call all other methods
        
        1. split train-val-test
        2. cont, cat split
        3. both continuous and categorical to sequential
        4. normalize
        5. if other model than LSTM flatten data

        """


def get_cont_cat(data, add_dates=False, cat_types=['int64']):
    num_cat = ContCatSplit(data, add_dates, cat_types)
    cont, cat = num_cat.cont_cat_split()
    return cont, cat


def get_loader(x, y, batch_size):
    # Return dict with {'features', 'targets'}
    return DataLoader(ToTorch(x, y), batch_size=batch_size)


def get_ts_split(data, train_size=0.75, valid_size=0.25, test_size=None):
    if test_size is not None:
        train_ds, valid_ds = ts_split(data, train_size, valid_size)
        valid_ds, test_ds = ts_split(valid_ds, train_size, test_size)
        return train_ds, valid_ds, test_ds
    train_ds, valid_ds = ts_split((data, train_size, valid_size))
    return train_ds, valid_ds


def _create_loders(data_splits, names):
    dataloaders = {}
    if isinstance(data_splits, tuple):
        for ts, name in zip(data_splits, names):
            x, y = create_rolling_ts(ts)
            dataloaders[f'{name}'] = get_loader(x, y)
        return dataloaders


def data_pipeline(
    data, 
    train_size, 
    valid_size, 
    test_size=None, 
    add_dates=True, 
    cat_types=['int64'],
    return_cont_cat=False
    ):
    """
    Return numerical and categorical loaders
    For combining numerical and categorical features 
    the model has to get as input different feature-loaders.
    e.g.

    continous_loader and categorical_loader

    Return a dict with dataloaders.
    """

    data_splits = ts_split(data, train_size, valid_size, test_size)

    names = None
    if test_size and len(data_splits) == 3:
        names = ['train', 'valid', 'test']
    else:
        names = ['train', 'valid']

    if return_cont_cat:
        cont, cat = ContCatSplit(data, add_dates, cat_types)
        continous_dataloaders = _create_loders(cont, names)
        categorical_dataloaders = _create_loders(cat, names)
        return continous_dataloaders, categorical_dataloaders
    else:
        if add_dates:
            data_splits = [add_datepart(x) for x in data_splits]
        dataloaders = _create_loders(data_splits, names)
        return dataloaders


def create_dummy_df():
    index = pd.date_range(start='2001-01-01', periods=500)
    arr = np.random.rand(500, 1)
    df = pd.DataFrame({'close': arr.reshape(-1)}, index=index)
    return df

if __name__ == '__main__':
    """Tests to do
    
    1) Test all methods...
    """
    np.random.seed(42)

    data = create_dummy_df()
    print('data.head():')
    print(data.head())
    train_ds, valid_ds = ts_split(raw_data=data)
    print('train_ds:')
    print(train_ds)
    print('valid_ds:')
    print(valid_ds)
    assert isinstance(train_ds, pd.core.frame.DataFrame)
    assert isinstance(ts_split(raw_data=data), tuple)


               

        




