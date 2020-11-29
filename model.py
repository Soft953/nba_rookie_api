from joblib import load
#import pickle

class Model:
    def __init__(self, model_path, scaler_path):
        try:
            self.model = load(model_path)
            self.scaler = load(scaler_path)
        except:
            raise ValueError('Cannot instantiate model or scaler, wrong path')

    def predict(self, features):
        if self.model is not None:
            return self.model.predict(features)
