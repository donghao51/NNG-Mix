from sklearn.neural_network import MLPClassifier

from myutils import Utils

class supervised():
    def __init__(self, seed:int, model_name:str=None):
        self.seed = seed
        self.utils = Utils()

        self.model_name = model_name
        self.model_dict = {'MLP':MLPClassifier}

    def fit(self, X_train, y_train, ratio=None):
        self.model = self.model_dict[self.model_name](random_state=self.seed)

        # fitting
        self.model.fit(X_train, y_train)

        return self

    def predict_score(self, X):
        score = self.model.predict_proba(X)[:, 1]
        return score