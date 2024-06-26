# About
- This project involves in building time series model using Long short-term memory (LSTM) that is  kind of RNN(Recurrent Neural Network)  on Tensorflow Framework in order to make prediction of future stock price movement pattern . To make price movement  smoother  and less volatile ,the model use [EMA Indicator](https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp#:~:text=The%20exponential%20moving%20average%20(EMA)%20is%20a%20technical%20chart%20indicator,importance%20to%20recent%20price%20data.)  instead price as input feature in the past 30 days to forecast future EMA values as output prediction over the next 5 days.
- We provide you with the End to End Solution from ingesting data into BigQuery to visualizing prediction result on Dashboard tool.
- This project is experimental project aims to familiarize me with the use of the Google Cloud platform in the areas of Machine Learning and Data analytics.
- This project was inspired by [TFEX-Trend-Prediction(2020)](https://github.com/technqvi/ML-StockPrice-TrendPrediction) to shift model from XGBoost to LSTM Time-Series
## Steps performed include the following task and process flow figure shown  below.
![process](https://github.com/technqvi/TimeSeries-LSTM-Forecast-PriceMovement/assets/38780060/55e5556a-f59a-45d2-8ffe-e6c560962aa0)

1. Load stock price data from Finance.yahoo.com to Bigquery.
2. Create technical analysis indicator such as EMA,MACD,SINGLA using [TA library](https://technical-analysis-library-in-python.readthedocs.io/en/latest/) as features and import data into the price data table.
3. Build Time Series LSTM Model.
   - Tune LSTM(RNN) model using Keras Tuner to find optimal hyperparameters to get the best model.
   - Save tuned model and scaler(1.feature scaler 2. prediction scaler) into google cloud storage.
4. Load tuned model to  make prediction and store prediction result into the prediction result table on BigQuery.
5. Visualize prediction result using line chart compared to the actual result through Jupyter Lab and PowerBI.
6. Collect model performance data such as Predicted Value and  Actual Value and bring them to calculate Mean Absolute Error(MAE) to monitor model performance over time.
7. Visualize error between predicted value and  actual value and show MAE value over time.
<img width="802" alt="report" src="https://github.com/technqvi/TimeSeries-LSTM-Forecast-PriceMovement/assets/38780060/184c06a0-5a8b-4c4c-8e49-e19cf210d8c7">

## [Youtube : Building LSTM Time-Series Models  to Predict Future Stock Price Movement](https://www.youtube.com/playlist?list=PLIxgtZc_tZWPCX4dAFJFhDPPGxEungxc8)

## Forecast Asset Future Price Movement By LSTM-TimeSeries Source Code
### [load_daily_price_from_yahoo.ipynb](https://github.com/technqvi/TimeSeries-LSTM-Forecast-PriceMovement/blob/main/load_daily_price_from_yahoo.ipynb)
##### Youtube : [1 Load Stock Price From Yahoo To BigQuery For Building LSTM Model](https://www.youtube.com/watch?v=jaPpyopNFPA&feature=youtu.be)
* There are 2 options to load price data to GoogleBiquery.
* Option#1 Export data price from Amibroker as csv file and load it to bigquery.
* Option#2 Pull data price from [finance.yahoo.com](https://finance.yahoo.com/) by using [yfinance](https://github.com/ranaroussi/yfinance) as dataframe and load it to bigquery .
* To build any technical analysis indicator as features to get ready for building Time-Series Machine Learning, we can apply [Technical](https://technical-analysis-library-in-python.readthedocs.io/en/latest/)[ Analysis](https://technical-analysis-library-in-python.readthedocs.io/en/latest/) Library in Python](https://technical-analysis-library-in-python.readthedocs.io/en/latest/) to get it done.
* This script has been deployed as a cloud function on google cloud run service AS [load-asset-price-yahoo](https://github.com/technqvi/TimeSeries-LSTM-Forecast-PriceMovement/tree/main/load-asset-price-yahoo)(google cloud function) and create a job on the cloud scheduler to trigger the cloud function to perform a defined task based on a daily basis.

### [build_forecast_ts_lstm_model.ipynb](https://github.com/technqvi/TimeSeries-LSTM-Forecast-PriceMovement/blob/main/build_forecast_ts_lstm_model.ipynb)
##### Youtube :  [2#1 Build Univariate Multi Step LSTM Models To Predict Stock Price](https://www.youtube.com/watch?v=O8p2cteVTSs&feature=youtu.be) | [2#2 Build Univariate Multi Step LSTM Models To Predict Stock Price](https://youtu.be/_bVOFtHC2yQ) |  [2#3 Build Univariate Multi Step LSTM Models To Predict Stock Price](https://www.youtube.com/watch?v=8idQEuBFLfw&feature=youtu.be)
* Load the training data from Big1uery  and save it as  csv file.
* Explore the data to identify trends and patterns of EMA movement.
* Split the data  into 2 parts such as  train and test dataset to prepare it for modeling.
* Normalize data by Scaling  data into given range 0-1(Min-Max).
* Tranform input feature from 2D array  to 3D  array  to feed it into the LSTM network .
* Build  model by Keras Tuner for tuning to find optimal hyper paramter to get best model.
* ReTrain with the best tuned model on the training data set.
* Evaluate with test dataset with selected regression metric to see how well model forecast  with metric MAE .
* Build final model with entire data .
* Store  model file and its scaler files into local path and GCS.


### [forecast_asset_movement.ipynb](https://github.com/technqvi/TimeSeries-LSTM-Forecast-PriceMovement/blob/main/forecast_asset_movement.ipynb)
##### Youtube :[3 Make Stock Multi Step Prediction Using LSTM Model](https://www.youtube.com/watch?v=8DlACgKslSE)
* Load model configuration metadata by model-id from csv file referenced as external table on BigQuery.
* Load model file and scaler file for feature and prediction value normalization.
* Check whether price data as specifed data on FinAssetForecast.fin_data table has been made prediction on FinAssetForecast.fin_movement_forecast table .
* Get the last N sequence records of specific features like EMA,MACD,SIGNAL as input feature from FinAssetForecast.fin_data table to make prediction of future  movement as prediction output. 
* Make prediction with proper input (3 dimension numpy array  [sample rows, time steps, features])
* Create 3 dataframes and covert Json file, there are 2 dataframe containing feature and prediction value  Feature Dataframe are contained as collection in  Main Dataframe .
* Ingest JSON file into FinAssetForecast.fin_movement_forecast table.
* This script has been deployed as clound function on google cloud as this link [forecast-asset-movement](https://github.com/technqvi/TimeSeries-LSTM-Forecast-PriceMovement/tree/main/forecast-asset-movement) and create job on cloud scheduler to trig clound function on daily basis.

### [invoke_forecast_gcf.ipynb](https://github.com/technqvi/TimeSeries-LSTM-Forecast-PriceMovement/blob/main/invoke_forecast_gcf.ipynb)
#### Youtube : [3 Make Stock Multi Step Prediction Using LSTM Model#2](https://youtu.be/8DlACgKslSE?t=4265)
* To make prediction multiple items , run this script to call cloud function api by specifying the desired period.
* To obtain authentication token as credential to call api correctly, you need to  install Google cloud-sdk and set default project first on environment variable(Window OS). [link](https://cloud.google.com/sdk/docs/install).


### [visualize_forecast_result.ipynb](https://github.com/technqvi/TimeSeries-LSTM-Forecast-PriceMovement/blob/main/visualize_forecast_result.ipynb)
#### YouTube: [4 Visualize Stock Price Prediction Result on JupyterLab](https://www.youtube.com/watch?v=jiOr3AIMWO4&)
* Specify start-date and end-date to plot prediction result
* Get model configuration from FinAssetForecast.model_ts_metadata table by the model id.
* Get feature and  prediction value from FinAssetForecast.fin_movement_forecast and actual value from  FinAssetForecast.fin_movement_forecast.
* Plot prediction result consisted of feature value and prediction value consecutively by comparison between prediction to actual price using line chart.
* Find mean absolute error(MAE) to measure gap between actual value and predicted value. 

###  [Prediction Result Analystics on PowerBI](https://app.powerbi.com/groups/me/reports/fa816185-f898-4b89-9d06-8864d39ec0eb/ReportSection?experience=power-bi)
#### YouTube: [4 Visualize Stock Price Prediction Result on PowerBI](https://youtu.be/jiOr3AIMWO4?t=2093)
* Show EMA features value series (blue dot) followed by EMA Prediction value (green dot).
Retrieve prediction result from view table in Bigquery.
Transform data in order to filter only EMA1 Feature.
Create Visualization prediction result(feature+prediction) compared to actual price with line chart on PowerBI.

### [collect_performance_forecast_result.ipynb](https://github.com/technqvi/TimeSeries-LSTM-Forecast-PriceMovement/blob/main/collect_performance_forecast_result.ipynb)
#### Youtub :[5 Collect&Monitor Time Series Model Performance Data](https://www.youtube.com/watch?v=Fd1GfmX_Z3k&list=PLIxgtZc_tZWPCX4dAFJFhDPPGxEungxc8&index=7)
* Create collection date on Saturday as well as start date and end date to gather model performance data every week.
* Get model configuration metadata ( This script is capable of collecting performance for multiple models once).
* Retrieve predicted value  and actual values  and return as dataframe from tables in BigQuery and merge both into one dataframe.
* Take the recently created dataframe as the first dataframe and  the previous the whole model performance data from table as the second dataframe to calculate MAE(Mean Absolute Error)  and put the calculation result into the dataframe.
* Convert dataframe to JSON object along with adding predicted value and actual values into this JSON object as nested and repeated columns and end up  loading this JSON object into BigQuery.
* Cloud function as this link [collect_performance_forecast_result](https://github.com/technqvi/TimeSeries-LSTM-Forecast-PriceMovement/tree/main/collect_performance_forecast_result) 


### Folder to store Artifact and other files
* [TimeSeriesML-FinMarket ](https://github.com/technqvi/TimeSeriesML-FinMarket) : This repository is used to  practice how to build the time series model with various methods like LSTM , ARIMA
* [model](https://github.com/technqvi/TimeSeries-LSTM-Forecast-PriceMovement/tree/main/model) :  Each sub folder stores model file and scaler object file, each is located on both local path and google cloud  storage.
* [model_ts_metadata.csv](https://github.com/technqvi/TimeSeries-LSTM-Forecast-PriceMovement/tree/main/model/model_ts_metadata.csv) : store model configuration metadata on google cloud storage but it allow you to query against external table on BigQuery.
* [train_data](https://github.com/technqvi/TimeSeries-LSTM-Forecast-PriceMovement/tree/main/train_data) : store train/test csv file loaded from Bigquery.
* [train_model_collection](https://github.com/technqvi/TimeSeries-LSTM-Forecast-PriceMovement/tree/main/train_model_collection) : All satisfactory models can be collected here.
* [tuning](https://github.com/technqvi/TimeSeries-LSTM-Forecast-PriceMovement/tree/main/tuning) : use Keras Tuner to find optimal hyperparamer and store tuning result here.
* [csv_data](https://github.com/technqvi/TimeSeries-LSTM-Forecast-PriceMovement/tree/main/csv_data) :To load data price exported from Amibrker to BQ, we will store this file here  .
* [data-schema-bq](https://github.com/technqvi/TimeSeries-LSTM-Forecast-PriceMovement/tree/main/data-schema-bq) : All table schema on Bigquery.
* [command to deploy script to cloud function](https://github.com/technqvi/TimeSeries-LSTM-Forecast-PriceMovement/tree/main/forecast-asset-deploy-function.txt)  

### [All Essential Packages  on Python 3.9](https://pypi.org/project)
- tensorflow >=2.11
- keras tuner >= 1.3
- scikit-learn >= 1.2.2
- pandas >=1.5.3 and numpy >= 1.24.2
- google-bigquery=3.7
- ta =0.10.2
- yfinance