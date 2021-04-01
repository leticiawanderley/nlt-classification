import numpy as np
import pandas as pd

from sklearn import preprocessing


def extract_sequence_into_separate_columns(dataframe, column):
    single_features = lambda x: pd.Series([i for i in (x.split(' ')
                                                       if isinstance(x, str)
                                                       else ['_', '_', '_'])])
    new_columns = dataframe[column].apply(single_features)
    new_columns.columns = [column + '_' + str(i) for i in new_columns.columns]
    new_columns = new_columns.replace(np.nan, '', regex=True)
    return new_columns


def encode_columns(dataframe):
    labels = []
    for col in dataframe.columns:
        labels += list(dataframe[col].unique())
    lb = preprocessing.LabelBinarizer()
    lb.fit(list(set(labels)))
    new_columns = {}
    for col in dataframe.columns:
        new_columns[col] = [int(''.join(list(map(str, x))), 2)
                            for x in lb.transform(dataframe[col])]
    return pd.DataFrame.from_dict(new_columns)


def create_dummies(dataframe):
    new_columns = []
    for col in dataframe.columns:
        dummies = pd.get_dummies(dataframe[col], prefix=col)
        new_columns.append(dummies)
    return pd.concat(new_columns, axis=1)
