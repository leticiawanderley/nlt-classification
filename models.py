from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def linear_model(x, y, x_test, y_test):
    reg = LogisticRegression(random_state=42).fit(x,y)
    return reg.score(x_test, y_test)


def random_forest(x, y, x_test, y_test):
    clf = RandomForestClassifier(n_estimators=400, max_depth=100, random_state=42)
    clf.fit(x, y)
    return clf.score(x_test, y_test)