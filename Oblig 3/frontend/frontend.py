import requests as r

inputs = {
     'Date': '2017-06-25',
     'Location': 'SydneyAirport',
     'MinTemp': 6.8,
     'MaxTemp': 19.5,
     'Rainfall': 0.0,
     'Evaporation': 3.4,
     'Sunshine': 9.4,
     'WindGustDir': 'WSW',
     'WindGustSpeed': 46.0,
     'WindDir9am': 'NW',
     'WindDir3pm': 0,
     'WindSpeed9am': 0,
     'WindSpeed3pm': 28.0,
     'Humidity9am': 67.0,
     'Humidity3pm': 42.0,
     'Pressure9am': 1018.4,
     'Pressure3pm': 1015.2,
     'Cloud9am': 1.0,
     'Cloud3pm': 1.0,
     'Temp9am': 11.9,
     'Temp3pm': 17.9,
     'RainToday': 'No'
}
url = 'http://127.0.0.1:5000/models/rf/predict'
response = r.get(url, params=inputs)
print(f'Response: {response} args: {response.json()})')

url = 'http://127.0.0.1:5000/models/rf/threshold'
print(r.get(url).json())
print(r.put(url, json={'threshold': 0.3}).json())
print(r.get(url).json())

url = 'http://127.0.0.1:5000/models/rf/predict'
response = r.get(url, params=inputs)
print(f'Response: {response} args: {response.json()})')