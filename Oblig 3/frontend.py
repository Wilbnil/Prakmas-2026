import requests as r

inputs = { #We should make a form
     'Date': '2026-11-11',
     'Location': 'Canberra',
     'MinTemp': 6,
     'MaxTemp': 20,
     'Rainfall': 50,
     'Evaporation': 20,
     'Sunshine': 10,
     'WindGustDir': 'N',
     'WindGustSpeed': 15,
     'WindDir9am': 'E',
     'WindDir3pm': 'NNE',
     'WindSpeed9am': 5,
     'WindSpeed3pm': 10,
     'Humidity9am': 20,
     'Humidity3pm': 50,
     'Pressure9am': 1015,
     'Pressure3pm': 1030,
     'Cloud9am': 2,
     'Cloud3pm': 3,
     'Temp9am': 8,
     'Temp3pm': 20,
     'RainToday': 'Yes'
}
url = 'http://127.0.0.1:5000/Models/mlp/predict'
response = r.get(url, params=inputs)
print(f'Response: {response} args: {response.json()})')

url = 'http://127.0.0.1:5000/Models/mlp/threshold'
print(r.get(url).json())
print(r.put(url, json={'threshold': 0.3}).json())
print(r.get(url).json())

url = 'http://127.0.0.1:5000/Models/mlp/predict'
response = r.get(url, params=inputs)
print(f'Response: {response} args: {response.json()})')