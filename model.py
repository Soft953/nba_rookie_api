from joblib import load

class Model:
    def __init__(self, path):
        try:
            self.model = load(path)
        except:
            raise ValueError('Cannot instantiate model, wrong path')

    def predict(self, features):
        if self.model is not None:
            return self.model.predict(features)
