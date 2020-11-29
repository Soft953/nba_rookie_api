from flask import Flask, request

from model import Model

app = Flask(__name__)


# Simple Flask API

@app.route('/')
def hello_world():
    try:
      model = Model(model_path='models/clf_knn.joblib', scaler_path='models/minmax_scaler.joblib')
      features_name = [
          'GP', 'MIN', 'PTS', 'FGM',
          'FGA', 'FG%', '3P Made', '3PA',
          '3P%', 'FTM', 'FTA', 'FT%', 'OREB',
          'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV'
      ]

      x = []
      for f in features_name:
          value = request.args.get(f)
          if value: x.append(value)
          else : x.append(0)

      x_minmax = model.scaler.transform([x])
      y_pred = model.model.predict(x_minmax)[0]
      return  "Prediction: " + str(y_pred) + ", so this player " + {0:'is not', 1:'is'}[y_pred] + " worth investing in NBA"
    except ValueError as e:
      return 'Error ' + str(e)
    return 'Hello, World!'


if __name__ == '__main__':
    app.debug = True
    app.run('0.0.0.0', port = 5000)