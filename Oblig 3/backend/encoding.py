import numpy as np
import pandas as pd

def cyclic_encoder(series, period):
    sin = np.sin(2 * np.pi * series / period)
    cos = np.cos(2 * np.pi * series / period)

    return sin, cos

location_features = ['Location']

rain_today_features = ['RainToday']

wind_features = [
    'WindGustDir',
    'WindDir9am',
    'WindDir3pm'
]

date_features = ['Date']
wind_map = {
    'N': 1, 'NNE': 2, 'NE': 3, 'ENE': 4,
    'E': 5, 'ESE': 6, 'SE': 7, 'SSE': 8,
    'S': 9, 'SSW': 10, 'SW': 11, 'WSW': 12,
    'W': 13, 'WNW': 14, 'NW': 15, 'NNW': 16
}

WIND_DIRECTIONS = 16
MONTH_PERIOD = 12

def encode_rain_today(rain_today_dataframe):
    return rain_today_dataframe.replace({'No': 0, 'Yes': 1}).astype(float)

def encode_wind(wind_data_frame):
    result = pd.DataFrame()

    for column in wind_data_frame.columns:

        wind_numbers = wind_data_frame[column].map(wind_map).astype(float)

        sin, cos = cyclic_encoder(wind_numbers, WIND_DIRECTIONS)
        result[column + '_Sine'], result[column + '_Cosine'] = sin, cos

    return result

def encode_date(dataframe):
    result = pd.DataFrame()

    date = pd.to_datetime(dataframe['Date'])
    year = date.dt.year
    month = date.dt.month
    day = date.dt.day_of_year

    is_leap_year = ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)
    days_in_year = np.select([is_leap_year], [366], default = 365)

    result['Month_Sine'], result['Month_Cosine'] = cyclic_encoder(month, MONTH_PERIOD)

    result['Day_Sine'], result['Day_Cosine'] = cyclic_encoder(day, days_in_year)

    result['Year'] = year

    return result
