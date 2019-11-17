# Traffic-Accident-Prediction
A real-time traffic accident predictor

## Data Extraction
Please refer to [`/data`](data/) to download and place them.  
Once the original dataset is downloaded, you can extract certain city's information.  
```
python utils/extract_city.py --city [City_Name] --state [State_Name] --filepath [Your Data Path]

ex.) python utils/extract_city.py --city "Palo Alto" --state "CA"
default city = "San Jose"
default state = "CA"
default filepath = "data/US_Accidents_May19.csv"
```

Once you run the above script, it will only extract accidents in a given city.  

## Training
This step will include training data generations (positive samples + negative samples) and training a classifier model (RandomForest)  

### Positive Sample Generations with DBSCAN
First thing is to cluster accidents with some hyperparameters.  
DBSCAN clustering algorithm is used to group accidents.
```
python utils/cluster.py --filepath [Your City Data Path] --cluster_distance [cluster distance] --cluster_samples [sample size]

ex.) python utils/cluster.py --cluster_distance 0.001 --cluster_samples 50
default filepath = "data/San Jose_CA.csv"
default cluster_distance = 0.001
default cluster_samples = 20
```
Please note that cluster distance is in terms of latitude degree. 0.001 degree in latitude is about 100 meters in metrics.  

### Negative Sample Generation
