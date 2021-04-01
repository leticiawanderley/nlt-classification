from itertools import combinations


def create_feature_combinations(features_list):
    combs = []
    for i in range(1, len(features_list)+1):
        for comb in combinations(features_list, i):
            combs.append(comb)
    return combs


def create_hyperparam_files(features):
    for i, feature_list in enumerate(features):
        f = open('features/input.' + str(i), 'w')
        for feature in feature_list:
            f.write(feature + '\n')
        f.close()


create_hyperparam_files(create_feature_combinations([
    'overall_score', 'exam_score', 'error_length',
    'error_type_dummies', 'error_type_encoded',
    'incorrect_deps_dummies', 'incorrect_deps_encoded',
    'incorrect_ud_tags_dummies', 'incorrect_ud_tags_encoded',
    'incorrect_ptb_tags_dummies', 'incorrect_ptb_tags_encoded',
]))
