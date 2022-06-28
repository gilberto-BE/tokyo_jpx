
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler


def set_date_index(df, col='Date'):
    df[col] = pd.to_datetime(df[col])
    df.set_index(col, inplace=True)
    return df


def get_data(folder='train_files'):
    """
    HOW TO HANDLE PREPROCESSING TEXT FOR 
    PREDICTION PIPELINE????

    Hard-coded rules for data.

    How it will be handle:
    Text transformer can be applied without saving the 
    state of the object, one reason, is that
    the categorical variables are static.

    Run the get data even for test data.
    """
    computer_name1 = 'gilbe'
    computer_name2 = 'Gilberto-BE'

    ROOT_PATH = f'c:/Users/{computer_name1}/Documents/TokyoData'
    train_df = pd.read_csv(f'{ROOT_PATH}/{folder}/stock_prices.csv')
    stock_list = pd.read_csv(f'{ROOT_PATH}/stock_list.csv').drop('Close', axis=1)

    TEXT_COLS = ['Section/Products', '33SectorName', '17SectorName', 'Universe0']
    stock_list = stock_list[TEXT_COLS + ['MarketCapitalization', 'SecuritiesCode']]
    train_df = stock_list.merge(train_df, on=['SecuritiesCode'])

    for txt_col in TEXT_COLS:
        train_df[txt_col] = TextTransform(list(train_df[txt_col])).transform()

    # # Add financials
    # df_financials = pd.read_csv(f'{ROOT_PATH}/train_files/financials.csv', low_memory=False)
    # df_financials.replace('－', np.nan, inplace=True)
    # df_financials.replace('NaN', np.nan, inplace=True)

    # FIN_COLS_CONT = [
    #     'NetSales', 'EquityToAssetRatio', 'TotalAssets', 'Profit', 
    #     'OperatingProfit', 'EarningsPerShare', 'Equity', 
    #     'BookValuePerShare', 'ResultDividendPerShare1stQuarter', 
    #     'ResultDividendPerShare2ndQuarter', 'ResultDividendPerShare3rdQuarter',
    #     'ResultDividendPerShareFiscalYearEnd', 'ResultDividendPerShareAnnual'
    #     ]
    # df_financials[FIN_COLS_CONT] = df_financials[FIN_COLS_CONT].astype(float)
    # df_financials = df_financials[FIN_COLS_CONT + ['SecuritiesCode', 'Date']]
    # train_df = train_df.merge(df_financials,  on=['SecuritiesCode', 'Date'], how='left')
    
    train_df = set_date_index(train_df)
    print('train_df.head(10):')
    print(train_df.head(10))


    # train_df['Date'] = pd.to_datetime(train_df['Date']) 
    # train_df.set_index('Date', inplace=True)


    return train_df


class TextTransform:
    def __init__(self, tokens):
        self.token_np = np.array(list(tokens)) #if isinstance(tokens, (pd.DataFrame, pd.Series)) else tokens
        self.vocab = np.unique(self.token_np)
        self.w2idx = {w: int(i) for i, w in enumerate(self.vocab)}
        self.idx2w = {int(i): w for i, w in enumerate(self.vocab)}

    def transform(self):
        self.idx = [self.w2idx[i] for i in self.token_np]
        return self.idx

    def inv_transform(self):
        return [self.idx2w[i] for i in self.idx]


def dataloader_by_stock(
    train_df, 
    sec_code, 
    batch_size=32,  
    continous_cols=['Close'],
    return_scaler=False,
    transform=StandardScaler
    ):
    df = train_df[train_df['SecuritiesCode'] == sec_code].drop(['SecuritiesCode'], axis=1)
    df = date_features(df)
    
    """

    TODO:
    1. CREATE A BRANCH FOR THE TESTING PHASE!!!!

    CHANGES HERE HAVE TO BE IMPLEMENTED IN dataloader_test_by_stock()
    Hard coded cat-columns
    
    
    """
    cat_cols = ['day_of_year', 'month', 'day_of_week', 'RowId', 'Section/Products', '33SectorName', '17SectorName']
    cont, cat = cont_cat_split(df, cat_cols=cat_cols)
    # print('continouscols:', cont.columns)
    
    print('continuos shape:', cont.shape, '', 'categorical shape:', cat.shape)
    
    df_train_cat, df_val_cat = ts_split(cat)
    # print('cat_columns:', df_train_cat.columns)
    df_train, df_val = ts_split(cont)

    xtrain, ytrain = preprocess(df_train, 'Target', 1, continous_cols=continous_cols)
    xval, yval = preprocess(df_val, 'Target', 1, continous_cols=continous_cols)

    # if transform is not None:
    #     scaler = transform()
    #     xtrain = scaler.fit_transform(xtrain)
    #     xval = scaler.transform(xval)

    train_loader = get_loader(
        x=xtrain, 
        y=ytrain, 
        batch_size=batch_size, 
        x_cat=df_train_cat.to_numpy()
        )
    val_dataloader = get_loader(
        x=xval, 
        y=yval, 
        batch_size=batch_size, 
        x_cat=df_val_cat.to_numpy()
        )
    # if return_scaler:
    #     return train_loader, val_dataloader, scaler
    return train_loader, val_dataloader


def dataloader_test_by_stock(
    train_df, 
    sec_code, 
    transformer=None, 
    batch_size=32,  
    continous_cols=['Close'],
    target_col='Target'
    ):
    df = train_df[train_df['SecuritiesCode'] == sec_code].drop(['SecuritiesCode'], axis=1)
    df = date_features(df)

    """Hard coded cat-columns"""
    cat_cols = ['day_of_year', 'month', 'day_of_week', 'RowId', 'Section/Products', '33SectorName', '17SectorName']


    # df['Target'] = df['Close'].shift().pct_change()
    # print(df.head())
    
    # cat_cols = ['day_of_year', 'month', 'day_of_week', 'RowId']
    cont, cat = cont_cat_split(df, cat_cols=cat_cols)
    
    # print('continuos shape:', cont.shape, '', 'categorical shape:', cat.shape)
    xtest = preprocess(cont, target_col, 1, continous_cols=continous_cols)

    # if transformer is not None:
    #     xtest = transformer.transform(xtest)
    # else:
    #     print('Notice that the transformer is None.')

    test_dataloader = get_predict_loader(
        x=xtest, 
        batch_size=batch_size, 
        x_cat=cat.to_numpy()
        )
    return test_dataloader


def ts_split(raw_data, train_size=0.85, val_size=None):
    """
    Hard code and train date.
    """
    train_sz = '2021-11-30' #int(len(raw_data) * train_size)
    train_set = raw_data[ :train_sz]
    valid_set = raw_data[train_sz: ]
    return train_set, valid_set    


def show_df(
    df, 
    show_info=True, 
    show_head=True, 
    show_tail=True, 
    dataframe_name='financials'
):
    print(f'<<< {dataframe_name} >>>')
    print(df.shape)
    if show_info:
        print(df.info())
    if show_head:
        print(df.head())
    if show_tail:
        print(df.tail())
    print('-.' * 80)
    
    
def date_features(df, date_col=None):
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)

    df.loc[:, 'day_of_year'] = df.index.dayofyear
    df.loc[:, 'month'] = df.index.month
    df.loc[:, 'day_of_week'] = df.index.day
    return  df


def preprocess(
    df, 
    target_col='Target',
    target_dim=1, 
    continous_cols=['Open', 'Close', 'High', 'Low', 'Volume']
    ):
    """
    -----------------
    Return x, y
    """
    rows = len(df)
    df['Volume'] = df['Volume'].astype(float)
    x = df.drop(target_col, axis=1) if target_col is not None else df
    x = x[continous_cols]
    for col in continous_cols:
        x[col] = x[col] * df['AdjustmentFactor']

    if continous_cols:
        x[continous_cols] = x[continous_cols].pct_change()
    x = x.select_dtypes(include=[int, float]).dropna().to_numpy()
    if target_col is not None:
        
        y = df[target_col].dropna()
        # y.plot()
        y = y.to_numpy().reshape(rows, target_dim)

        return x, y
    else:
        # x = df
        return x


def cont_cat_split(df, col_type=None, cat_cols=None):
    """
    Return transformer!!!
    """
    if cat_cols:
        cat = df[cat_cols]
        if 'RowId' in cat_cols:
           enc = OrdinalEncoder()
           # Transform to int??
           txt_transfom = TextTransform(cat['RowId'])
           cat.loc[:, ['RowId']] = txt_transfom.transform()
    elif col_type is not None:
        cat = df.select_dtypes(include=col_type)
    cat_cols = cat.columns
    cont = df.drop(cat_cols, axis=1)
    return cont, cat


class ToTorch(Dataset):

    def __init__(
            self,
            num_features,
            target,
            cat_features=None
            ):
        self.num_features = num_features
        self.target = target
        self.cat_features = cat_features

    def __len__(self):
        return len(self.num_features)

    def __getitem__(self, idx):
        num_features = self.num_features[idx]
        target = self.target[idx]
        cat_features = self.cat_features[idx]

        if self.cat_features is not None:
            return {
                'num_features': torch.from_numpy(np.array(num_features)).float(), 
                'target': torch.from_numpy(np.array(target)).float(),
                'cat_features': torch.from_numpy(np.array(cat_features)).int()
                }

        return {
            'num_features': torch.from_numpy(np.array(num_features)).float(), 
            'target': torch.from_numpy(np.array(target)).float()
            }
    

def get_loader(x, y, batch_size, x_cat=None):
    # Return dict with {'num_features', 'targets'}
    if x_cat is not None:
        return DataLoader(ToTorch(x, y, x_cat), batch_size=batch_size)
    return DataLoader(ToTorch(x, y), batch_size=batch_size)


class TestLoader(Dataset):

    def __init__(
            self,
            num_features,
            cat_features=None
            ):
        self.num_features = num_features
        self.cat_features = cat_features

    def __len__(self):
        return len(self.num_features)

    def __getitem__(self, idx):
        num_features = self.num_features[idx]
        # target = self.target[idx]
        cat_features = self.cat_features[idx]

        if self.cat_features is not None:
            return {
                'num_features': torch.from_numpy(np.array(num_features)).float(), 
                'cat_features': torch.from_numpy(np.array(cat_features)).int()
                }

        return {
            'num_features': torch.from_numpy(np.array(num_features)).float(), 
            }

def get_predict_loader(x, batch_size, x_cat=None):
    if x_cat is not None:
        return DataLoader(TestLoader(x, x_cat), batch_size=batch_size)
    return DataLoader(TestLoader(x), batch_size=batch_size)


if __name__ == '__main__':
    token_list = ['Prime Market', 'Prime Market', 'two', 'three', 'four', 'four', 'four']

    # token_list = [sent_tokenize(token_list)]
    print('raw tokens')
    print(token_list)
    txt_trans = TextTransform(token_list)
    print('Get indexes')
    print(txt_trans.transform())
    print()
    print('get dict with vocab')
    print(txt_trans.w2idx)
    print()
    print('inverse transform')
    print(txt_trans.inv_transform())
