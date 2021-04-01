import pandas as pd
import spacy

from constant import DATA_FIELDS


PIPE = '|'
SPACED_PIPE = ' | '
PIPE_AT_POS_0 = '| '
EMPTY = ''
SPACE = ' '


def create_feature_string(features):
    if not features:
        return ' '
    return ' '.join(features)


def read_data(filename):
    df = pd.read_csv(filename)
    df = df[DATA_FIELDS]
    df = df[df['Negative transfer?'].isin(['Y', 'N'])]
    return df


def create_ngrams(feature_list, index):
    features = feature_list[index:index+3]
    if len(features) < 3:
        features += ['_ ' * (3 - len(features))]
    return features


def extract_linguistic_features(nlp, sentence):
    if sentence.isupper():
        sentence = sentence.lower()
    ptb_tags = []
    ud_tags = []
    deps = []
    index = None
    for sent in nlp.pipe([sentence], disable=["ner", "textcat"]):
        for i, token in enumerate(sent):
            if token.text == PIPE and index is None:
                index = i
            else:
                ptb_tags.append(token.tag_)
                ud_tags.append(token.pos_)
                deps.append(token.dep_)
    return {'ptb_tags': ptb_tags, 'ud_tags': ud_tags,
            'deps': deps}, index


def extract_ngrams(nlp, dataframe, column):
    ngrams_dict = {
        'ptb_tags': [],
        'ud_tags': [],
        'deps': [],
    }
    for index, row in dataframe.iterrows():
        features_dict, index = \
            extract_linguistic_features(nlp, row[column])
        for key in features_dict:
            features = create_ngrams(features_dict[key], index)
            ngrams_dict[key].append(
                create_feature_string(features))
    return ngrams_dict


def process_fce_data(filenames):
    df = pd.DataFrame()
    for filename in filenames:
        df = pd.concat([read_data(filename), df]).reset_index(drop=True)
    nlp = spacy.load('en_core_web_lg')
    incorrect_ngrams_dict = extract_ngrams(nlp, df, 'incorrect_sentence')
    for key in incorrect_ngrams_dict:
        df['incorrect_'+key] = incorrect_ngrams_dict[key]
    df.to_csv('data/processed_data.csv')


process_fce_data(['data/new_chinese.csv', 'data/zhs_unsure.csv'])
