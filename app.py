from flask import Flask, request

from model import Model

app = Flask(__name__)


# Simple Flask API

@app.route('/')
def predict():
    """Return prediction based on request.args

    Example: GET request args should look like

    {
        'GP': 36,
        'MIN': 27.4,
        'PTS': 7.4,
        'FGM': 2.6,
        'FGA': 7.6,
        'FG%': 34.7,
        '3P Made': 0.5,
        '3PA': 2.1,
        '3P%': 25,
        'FTM': 1.6,
        'FTA': 2.3,
        'FT%': 69.9,
        'OREB': 0.7,
        'DREB': 3.4,
        'REB': 4.1,
        'AST': 1.9,
        'STL': 0.4,
        'BLK': 0.4,
        'TOV': 1.3
    }

    Returns:
        str: prediction, a player is worth investing or not
    """
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