from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

model = pickle.load(open('df.pkl', 'rb'))

@app.route('/')
def main():
    return render_template('dashboard.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json  

    gongsil = data['gongsil']
    profit = data['profit']
    longitude = data['longitude']
    latitude = data['latitude']
    guimo = data['guimo']
    floor = data['floor']


    input_data = pd.DataFrame({
        '공실률': [gongsil],
        '순영업소득' : [profit],
        '경도': [longitude],
        '위도': [latitude],
        '규모_소규모': [1 if guimo == '소규모' else 0],
        '규모_중대형': [1 if guimo == '중대형' else 0],
        '규모_집합': [1 if guimo == '집합' else 0],
        '층_1층': [1 if floor == '1층' else 0],
        '층_2층': [1 if floor == '2층' else 0],
        '층_3층': [1 if floor == '3층' else 0],
        '층_4층': [1 if floor == '4층' else 0],
        '층_5층': [1 if floor == '5층' else 0],
        '층_6층이상': [1 if floor == '6층이상' else 0],
        '층_지하1층': [1 if floor == '지하1층' else 0],
    })
    
    input_array = np.array(input_data)

    prediction = model.predict(input_array)

    prediction_float = prediction.item()

    response = {
        'prediction': prediction_float
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)