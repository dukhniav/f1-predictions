import datetime
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def convert_time(time_str):
    try:
        minutes, seconds = time_str.split(":")
        return float(minutes) * 60 + float(seconds)
    except:
        pass


# Define the API endpoint and parameters
url = 'https://ergast.com/api/f1/'
season = '2021'
round_num = '1'
season_offset = 5

# Get the results from the API for the last 5 races or last season's races if current round is less than 5
if int(round_num) <= season_offset:
    endpoint = url + str(int(season) - 1) + '/results.json'
    response = requests.get(endpoint)
    data = response.json()[
        'MRData']['RaceTable']['Races'][-(season_offset - int(round_num) + 1):]
else:
    endpoint = url + season + '.json?limit=' + \
        str(season_offset) + '&offset=' + \
        str(int(round_num) - season_offset - 1)
    response = requests.get(endpoint)
    data = response.json()['MRData']['RaceTable']['Races']


if response.status_code != 200:
    print(f"API request failed with status code {response.status_code}")
    exit()

try:
    data = response.json()['MRData']['RaceTable']['Races']
except:
    print("Failed to parse response as JSON")
    exit()

dfs = []
for i, race in enumerate(data):
    race_data = race['Results']
    df = pd.json_normalize(race_data)
    df['position'] = pd.to_numeric(df['position'])
    df['grid'] = pd.to_numeric(df['grid'])
    df['laps'] = pd.to_numeric(df['laps'])
    df['status'] = pd.Categorical(df['status'])
    df['driver.nationality'] = pd.Categorical(df['Driver.nationality'])
    df['constructor.name'] = pd.Categorical(df['Constructor.name'])
    dfs.append(df)

# Concatenate the preprocessed data from all races into a single dataframe for training and prediction
df = pd.concat(dfs)

# Define the features and the target variable
X_train = df[['grid', 'laps', 'status',
              'driver.nationality', 'constructor.name']]
y_train = df['position']

# Convert categorical features to one-hot encoding
X_train = pd.get_dummies(
    X_train, columns=['status', 'driver.nationality', 'constructor.name'])

# Define the model and train it on the historical data
model = LinearRegression()
model.fit(X_train, y_train)

# Define the API endpoint and parameters for the upcoming race qualifying
round_num = '6'  # Update with the current race round number
endpoint = url + season + '/' + round_num + '/qualifying.json'

# Get the data from the API and preprocess it
response = requests.get(endpoint)
data = response.json()['MRData']['RaceTable']['Races'][0]['QualifyingResults']
df = pd.json_normalize(data)
df['position'] = pd.to_numeric(df['position'])
df['q1'] = df['Q1'].apply(convert_time)
df['q2'] = df['Q2'].apply(convert_time)
df['q3'] = df['Q3'].apply(convert_time)
df['driver.nationality'] = pd.Categorical(df['Driver.nationality'])
df['constructor.name'] = pd.Categorical(df['Constructor.name'])

# Define the features and the target variable for the qualifying data
X_q = df[['q1', 'q2', 'q3', 'driver.nationality', 'constructor.name']]
X_q = pd.get_dummies(X_q, columns=['driver.nationality', 'constructor.name'])

X_q['grid'] = 0  # Set default grid position to 0
X_q['laps'] = 0  # Set default number of laps to 0
X_q['status_+1 Lap'] = 0  # Set default status to 0 for missing status columns
X_q['status_Accident'] = 0
X_q['status_Brakes'] = 0
X_q['status_Finished'] = 0
X_q['status_Retired'] = 0

# reorder columns of X_q to match X_train
X_q = X_q[X_train.columns]


# Predict the qualifying results using the updated model
y_pred = model.predict(X_q)

# Create a new DataFrame with driver names and predicted positions
drivers = df['Driver.code']
predicted_positions = pd.Series(y_pred, name='predicted_position')
predicted_df = pd.concat([drivers, predicted_positions], axis=1)

# Print the predicted qualifying positions with driver names
print('Predicted qualifying positions:')
print(predicted_df.sort_values('predicted_position'))
