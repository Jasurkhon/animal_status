import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    tracker_id = data['tracker_id']
    
    created_at = data['created_at']
    date_format = '%Y-%m-%d'
    created_at = datetime.strptime(created_at, date_format)
    created_at = created_at.date()

    battery_level = int(data['battery_level'])
    steps_count = int(data['steps_count'])
    animal_type = data['animal_type']

    #loading datasets
    tracker_data = pd.read_csv('animals.csv', sep=",")
    animal_types = pd.read_csv('animal_types.csv', sep=";")

    #merging datasets on tracker_id
    merged_data = pd.merge(tracker_data, animal_types, on='tracker_id')

    #if missing values yes or no
    if merged_data.isnull().sum().any():
        merged_data = merged_data.dropna()

    #created_at to a datetime
    merged_data['created_at'] = pd.to_datetime(merged_data['created_at'], format='mixed')

    #feature engineering

    #grouping by tracker_id, date, and calculating the avg battery level and steps count
    daily_data = merged_data.groupby(['tracker_id', merged_data['created_at'].dt.date]).agg({'battery_level': 'mean', 'steps_count': 'sum'}).reset_index()

    #normalization
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(daily_data[['battery_level', 'steps_count']])
    daily_data[['battery_level', 'steps_count']] = scaled_features

    #features for clustering
    features = ['battery_level', 'steps_count']

    #K-means clustering

    #determining number of clusters
    k_values = range(2, 10)
    inertia = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(daily_data[features])
        inertia.append(kmeans.inertia_)
    
    #after analysis got for k value 3
    optimal_k = 3 

    kmeans = KMeans(n_clusters=optimal_k, random_state=0)
    kmeans.fit(daily_data[features])

    #cluster labels
    daily_data['cluster_label'] = kmeans.labels_

    #centera of each cluster
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

    #calculating a sickness score for each animal and day
    daily_data['sickness_score'] = daily_data.apply(lambda row: ((row['battery_level'] - cluster_centers[row['cluster_label']][0]) ** 2) + ((row['steps_count'] - cluster_centers[row['cluster_label']][1]) ** 2), axis=1)

    #calculating mean sickness threshold
    mean_sickness_threshold = daily_data['sickness_score'].mean()

    #calculating standard deviation sickness score
    std_dev_sickness_score = daily_data['sickness_score'].std()

    #sickness threshold to determine if animal is sick or not 
    sickness_threshold = mean_sickness_threshold + (2 * std_dev_sickness_score)

    #creating a new column showing if the animal is sick or not for each day
    daily_data['is_sick'] = daily_data['sickness_score'] > sickness_threshold

    daily_data['created_at'] = pd.to_datetime(daily_data['created_at'])

    #getting only one value per day for animal health
    aggregated_data = daily_data.groupby(['tracker_id', daily_data['created_at'].dt.date])['is_sick'].max().reset_index()

    #renaming "True" and "False" values as "sick" and "not sick"
    aggregated_data['is_sick'].replace({True: 'sick', False: 'not sick'}, inplace=True)

    #Result
    result = aggregated_data[['tracker_id', 'created_at', 'is_sick']]
    
    animal_data = result[(result['tracker_id'] == tracker_id) & (result['created_at'] == created_at)][['tracker_id','created_at','is_sick',]].to_dict(orient='records')
    
    # Return the result as JSON
    return jsonify(animal_data)

if __name__ == '__main__':
    app.run()  