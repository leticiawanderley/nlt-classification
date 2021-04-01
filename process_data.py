import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from constant import PROCESSED_DATA_FIELDS
from utils import create_dummies, encode_columns,\
                  extract_sequence_into_separate_columns


def process_dataframe(filename):
    df = pd.read_csv(filename, index_col=[0])
    df['Negative transfer?'] = np.where(df['Negative transfer?'] == 'Y', 1, 0)
    df['exam_score'] = np.where(df['exam_score'] == '2.3T', '2.3',
                                df['exam_score'])
    df['exam_score'] = df['exam_score'].apply(pd.to_numeric)
    categorical_columns = [
        'error_type', 'incorrect_ptb_tags', 'incorrect_ud_tags',
        'incorrect_deps']

    feature_dict = {}
    for column in categorical_columns:
        separate_columns = extract_sequence_into_separate_columns(df, column)
        dummies = create_dummies(separate_columns)
        encoded = encode_columns(separate_columns)
        feature_dict[column + '_dummies'] = list(dummies.columns)
        feature_dict[column + '_encoded'] = list(encoded.columns)
        df = pd.concat([df, dummies, encoded], axis=1)
    return df, feature_dict


def create_train_test_split(dataframe):
    x_train, x_test, y_train, y_test = train_test_split(
        dataframe, dataframe['Negative transfer?'],
        test_size=0.1, random_state=42)
    x_train.to_csv('./data/x_train.csv')
    x_test.to_csv('./data/x_test.csv')
    y_train.to_csv('./data/y_train.csv')
    y_test.to_csv('./data/y_test.csv')


def save_feature_dict(any_dict):
    json_dict = json.dumps(any_dict)
    f = open('data/feature_dict.json', 'w')
    f.write(json_dict)
    f.close()


def main():
    df, feature_dict = process_dataframe('./data/processed_data.csv')
    feature_fields = [field for features in feature_dict.values() for field in features]
    create_train_test_split(df[PROCESSED_DATA_FIELDS + feature_fields])
    save_feature_dict(feature_dict)


main()
