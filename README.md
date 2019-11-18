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
Until this step, we only have positive samples (accidents). However, in order to train a classifier model, we need to have negative samples (no accidents) to distinguish accidents. Therefore, our script will generate 3x more negative samples per cluster. For instance, if a cluster has 200 accidents, 600 random negative samples will be generated.  
In order for an accurate negative sample generation, weather API (DarkSky API) is used. The date and hour will be randomly selected that spans in between the period of the positive samples. Then, it will also check whether my randomly generated date is not colliding with a positive sample's date and hour.
In order to use weather API, please sign up and get an API key from [DarkSkyAPI](https://darksky.net/dev). Please note that DarkSky API allows only 1000 free API calls per day. For San Jose, CA, about 34,000 API calls are needed to generate total negative samples. Please consult the pricing from DarkSky API before generating negative samples. You can access to San Jose, CA training set [here](https://drive.google.com/file/d/15v3SL2thC_H_iMjCXnaYohKewoztFyfU/view?usp=sharing).  
Since API calls are not that fast, it may take few hours to complete generating negative samples

```
python utils/negative_samples.py --filepath [Your Positive Sample Data Path] --apipath [Your Weather API text file]

ex.) python utils/negative_samples.py
default filepath = "data/PositiveTrainingData.csv"
default apipath = "api/darksky_api.txt"
```

Once you complete the negative samples generation, you will have `training.csv` in `data/` directory.

### Training a classifier
For this project, random forest is used to classify wether given some input conditions, there is an accident or not.
```
pthon utils/train_classifier.py --filepath [Your training data path] --train_by_cluster [Whether you want to train by cluster or not]

ex.) python utils/train_classifier.py
default fileapth = "data/training.csv"
default train_by_cluster = False
```

You can set the `train_by_cluster = True` if you want to train your data regardless of clusters. In other words, the classifier will activate entire hotspots (clusters) if it predicts there will be an accident. The final classifier model will be saved in `models/`  
If you set `train_by_cluster = False`, you are training cluster by cluster. Every cluster's classifier will be saved in `models/classifier_by_cluster`.

## Inference
Becuase the inference model is more intuitive with Google Map API, I suggest you to obtain Google Map API key [here](https://developers.google.com/maps/documentation/javascript/get-api-key). Also, it requires actual weather information for the location, please register for the weather API on DarkSky API. If you are willing to get the nearest accident hotspot from your location, you can run the following script.

```
python inference.py --single_mode [Whether you are predicting in a single mode] --lat [your location latitude] --lng [your location longitude]

ex.) python inference.py --single_mode False --lat 37.23 --lng -121.74
default single_mode = True
default latitude = 37.336431
default longitude = 121.883980
```
I highly recommend using Jupyer notebook for this application since it is more intuitive watching steps with Google Map API.

## Summary
This project was completed for CMPE 255 (Data Mining) class in Fall 2019 at San Jose State University.