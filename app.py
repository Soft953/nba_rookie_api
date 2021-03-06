from flask import Flask, request, jsonify
import yaml

from model import Model
from exception import InvalidUsage

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

        with open('configs.yaml', 'r') as f:
            configs = yaml.safe_load(f)

        if 'api' in configs and 'model_path' in configs['api']:
            model_path = configs['api']['model_path']
        if 'api' in configs and 'scaler_path' in configs['api']:
            scaler_path = configs['api']['scaler_path']

        model = Model(model_path=model_path, scaler_path=scaler_path)
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
        raise InvalidUsage(str(e), status_code=500)
    except yaml.YAMLError as e:
        raise InvalidUsage(str(e), status_code=500)
    except Exception as e:
        raise InvalidUsage(str(e), status_code=500)


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


if __name__ == '__main__':
    app.debug = True
    app.run('0.0.0.0', port = 5000)