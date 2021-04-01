import argparse
import csv
import json
import pandas as pd
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


warnings.simplefilter(action='ignore', category=FutureWarning)


def cross_validation_step(x_train, y_train, columns):
    lg_scores = cross_val_score(LogisticRegression(random_state=42),
                                x_train[columns], y_train['Negative transfer?'], cv=10)
    rf_scores = cross_val_score(RandomForestClassifier(n_estimators=400,
                                max_depth=100, random_state=42),
                                x_train[columns], y_train['Negative transfer?'], cv=10)
    return lg_scores, rf_scores


def main(input_file):
    input_columns = [col.replace('\n', '')
                     for col in open(input_file, 'r').readlines()]
    features_dict = json.load(open('./data/feature_dict.json', 'r'))
    columns = []
    for column in input_columns:
        if column in features_dict:
            columns.extend(features_dict[column])
        else:
            columns.append(column)
    train_x = pd.read_csv('./data/x_train.csv')
    train_x['error_length'] = train_x['error_length'].fillna(0)
    train_y = pd.read_csv('./data/y_train.csv')
    lg_scores, rf_scores = cross_validation_step(train_x, train_y, columns)
    with open('./data/results_tuning.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([' '.join(input_columns),
                         str(lg_scores.mean()), str(rf_scores.mean())])


def parse_arg_list():
    """Uses argparse to parse the required parameters"""
    parser = argparse.ArgumentParser(
                description='',
                formatter_class=argparse.RawTextHelpFormatter)
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument(
        '-f', '--input_file', help='File that contains input parameters')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arg_list()
    main(args.input_file)
