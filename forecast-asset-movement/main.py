#!/usr/bin/env python
# coding: utf-8

# In[110]:


import pandas as pd
import numpy as np
import os
from datetime import datetime,date,timedelta,timezone
import pytz
import json



from tensorflow.keras.models import load_model
import joblib

from google.cloud import storage
from google.cloud import bigquery
from google.oauth2 import service_account
from google.cloud.exceptions import NotFound
from google.api_core.exceptions import BadRequest


# In[111]:


# uncomment the following section and chage mode to gcs
# fuction
# json and env if/else
# return statment

import functions_framework
@functions_framework.http
def forecast_asset_movement(request):


    # # Parameter

    loadModelMode='gcs'   # local,gcs

    model_id='spy-ema1-60t10-ds0115t0523'
    today=''
    input_sequence_length =30
    output_sequence_length =5

    if request.get_json():
        print("JSON Post Date Info") # Post Method
        request_json = request.get_json()
        today=request_json['TODAY']
        model_id=request_json['MODEL_ID']

    else:
        print("Enviroment Variable Info")
        today=os.environ.get('TODAY', '') 
        #today=os.environ.get('TODAY', '2023-04-28')  
        model_id=os.environ.get('MODEL_ID', 'spy-ema1-60t10-ds0115t0523')  
  

    print("List parameter as belows")
    print(f"today={today}")
    print(f"model_id={model_id}")



    # # BigQuery Setting

    # In[113]:


    projectId='pongthorn'
    dataset_id='FinAssetForecast'
    table_id = f"{projectId}.{dataset_id}.fin_movement_forecast"
    table_data_id = f"{projectId}.{dataset_id}.fin_data"
    table_model_id= f"{projectId}.{dataset_id}.model_ts_metadata"

    print(table_id)
    print(table_data_id)
    print(table_model_id)

    # json_credential_file=r'C:\Windows\pongthorn-5decdc5124f5.json'
    # credentials = service_account.Credentials.from_service_account_file(json_credential_file)
    # client = bigquery.Client(project=projectId,credentials=credentials )
    client = bigquery.Client(project=projectId )

    def load_data_bq(sql:str):
     query_result=client.query(sql)
     df=query_result.to_dataframe()
     return df


    # # Load Model MetaData Configuration

    # In[114]:


    sqlModelMt=f"""
    SELECT * FROM `{table_model_id}`  where model_id='{model_id}'
    """
    print(sqlModelMt)
    dfModelMeta=load_data_bq(sqlModelMt)
    if dfModelMeta.empty==False:
        modelMeta=dfModelMeta.iloc[0,:]
        asset_name=modelMeta['asset']
        prediction_name=modelMeta['prediction']

        input_sequence_length=int(modelMeta['input_sequence_length'])
        output_sequence_length=int(modelMeta['output_sequence_length'])

        model_path=modelMeta['gs_model_path']
        local_model_path=modelMeta['local_model_path']

        model_file=modelMeta['model_file']
        scaler_file=modelMeta['scaler_file']
        scalerPred_file=modelMeta['scaler_pred_file']
        print(modelMeta)

    else: 
        raise Exception(f"Not found model id  {model_id}")



    # print(f"{today} - {asset_name} -{prediction_name}")
    # print("==================================================")


    # # Load model and scaler

    # In[115]:


    if loadModelMode=='local':
     objectPaht=local_model_path
    else:
     objectPaht=model_path  

    model_path=f"{objectPaht}/{model_file}"
    scale_input_path=f"{objectPaht}/{scaler_file}"
    scale_output_path=f"{objectPaht}/{scalerPred_file}"

    print(f"load model from {loadModelMode}")   
    print(model_path)
    print(scale_input_path)
    print(scale_output_path)



    # In[116]:


    if loadModelMode=='local':
        try:
            print("Model and Scaler Object Summary")
            x_model = load_model(model_path)
        except Exception as ex:
            print(str(ex))
            raise Exception(str(ex)) 

        try:
            print("Scaler Max-Min")
            x_scaler = joblib.load(scale_input_path)
            x_scalerPred=joblib.load(scale_output_path)

        except Exception as ex:
            print(str(ex))
            raise Exception(str(ex))

        print("=====================================================================================================")


    # In[117]:


    if loadModelMode=='gcs':
     try:    
        gcs_client = storage.Client()

        with open(scaler_file, 'wb') as scaler_f, open(scalerPred_file, 'wb') as scaler_pred_f,open(model_file, 'wb') as model_f:
            gcs_client.download_blob_to_file(scale_input_path, scaler_f
            )
            gcs_client.download_blob_to_file(scale_output_path, scaler_pred_f
            )
            gcs_client.download_blob_to_file(model_path, model_f
            )

        x_scaler = joblib.load(scaler_file)
        x_scalerPred=joblib.load(scalerPred_file)
        x_model = load_model(model_file)
     except Exception as ex:
        print(str(ex))
        raise Exception(str(ex))


    # In[118]:


    print(x_model.summary())
    #(max - min) / (X.max(axis=0) - X.min(axis=0))
    print(f"max={x_scaler.data_max_} and min={x_scaler.data_min_} and scale={x_scaler.scale_}")
    print(f"max={x_scalerPred.data_max_} and min={x_scalerPred.data_min_} and scale={x_scalerPred.scale_}")


    # # Declare and Initialize TS Model Variable

    # In[119]:


    date_col='Date'
    prediction_col=prediction_name
    feature_cols=[prediction_name]



    nLastData=input_sequence_length*2

    # dt_imported=datetime.now()
    dt_imported=datetime.now(timezone.utc)
    dtStr_imported=dt_imported.strftime("%Y-%m-%d %H:%M:%S")
    print(dtStr_imported)


    # # Query Fin Data from BQ

    # In[120]:


    lastDate=None
    if today=='':
        sqlLastDate=f""" select max(Date) as LastDate  from `{table_data_id}` where Symbol='{asset_name}' """

    else:
        sqlLastDate=f""" 
        select Date as LastDate  from `{table_data_id}` where Symbol='{asset_name}' 
        and Date='{today}'
        """
    print(sqlLastDate)
    results = client.query(sqlLastDate)
    dfLastDate=results.to_dataframe()
    print(dfLastDate)
    if dfLastDate.empty:
        print( f"Not found price data at {today}  of {asset_name}")
        return f"Not found price data at {today}  of {asset_name}"
    else:
        lastDate=dfLastDate.iloc[0,0]
        today=lastDate.strftime('%Y-%m-%d')


    print(f"Forecast {prediction_col} movement of  {asset_name} at {today}")


    # In[121]:


    print(f"Get last price of {asset_name}")
    sqlLastPred=f"""select prediction_date,asset_name,prediction_name,pred_timestamp from `{table_id}` 
    where prediction_date='{today}' and   asset_name='{asset_name}' and prediction_name='{prediction_col}'
    order by pred_timestamp 
    """
    print(sqlLastPred)
    dfLastPred=load_data_bq(sqlLastPred)
    if dfLastPred.empty==False:
       dfLastPred=dfLastPred.drop_duplicates(subset=['prediction_date','asset_name','prediction_name'],keep='last') 
       print(f"{asset_name}-{prediction_col}-{today} has been predicted price movement")
       print(dfLastPred)
       return f"Prediction price movement of {asset_name}-{prediction_col} at {today} has been predicted"
    else:
       print(f"{asset_name}-{prediction_col} at {today} has not been predicted price movement yet.") 
       print("The system is about to predict price movement shortly.") 
       print("=======================================================================================") 


    # In[122]:


    dayAgo=datetime.strptime(today,'%Y-%m-%d') +timedelta(days=-nLastData)
    print(f"Get data from {dayAgo.strftime('%Y-%m-%d')} - {today} as input to forecast")


    # In[123]:


    sql=f"""
    SELECT  *  FROM `{table_data_id}`  
    Where  {date_col} between  DATE_SUB(DATE '{today}', INTERVAL {nLastData} DAY) 
    and '{today}' and Symbol='{asset_name}' order by {date_col},ImportDateTime
    """
    print(sql)
    query_result=client.query(sql)
    df=query_result.to_dataframe()

    df=df.drop_duplicates(subset=[date_col,'Symbol'],keep='last')
    df[date_col]=pd.to_datetime(df[date_col],format='%Y-%m-%d')
    df.set_index(date_col,inplace=True)

    print(df.info())

    print(df[['Symbol','Close' ,'ImportDateTime']].head())
    print(df[['Symbol','Close' ,'ImportDateTime']].tail())

    if df.empty==True or len(df)<input_sequence_length:
        print(f"There is no enough data to make prediction during {dayAgo.strftime('%Y-%m-%d')} - {today}")
        return f"There is no enough data to make prediction during {dayAgo.strftime('%Y-%m-%d')} - {today}"


    # In[124]:


    # import matplotlib.pyplot as plt
    # import matplotlib.dates as mdates
    # import seaborn as sns

    # plt.subplots(2, 1, figsize = (20, 10),sharex=True)

    # ax1 = plt.subplot(2, 1, 1)
    # plt.plot(df[['Close','EMA1','EMA2']])
    # plt.ylabel('Price & EMA')

    # ax2 = plt.subplot(2, 1, 2)
    # plt.plot(df[['MACD','SIGNAL']])
    # plt.xlabel('Date')
    # plt.ylabel('MACD & Signal')

    # plt.show()


    # # Get only Feature( 1 Indicator) to Predict itself in the next N days

    # In[125]:


    print(f"Get Feature to Predict : {prediction_col} ")
    dfForPred=df[feature_cols]
    #dfForPred=dfForPred.iloc[-(input_sequence_length+1):-1,:]
    dfForPred=dfForPred.iloc[-input_sequence_length:,:]
    print(dfForPred.info())
    print(dfForPred.shape)

    print(dfForPred.head(5))
    print(dfForPred.tail(5))

    # dfForPred.plot(figsize = (20, 10))
    # plt.show()


    # # Make Pediction as Forecast

    # In[126]:


    xUnscaled=dfForPred.values #print(xUnscaled.shape)
    xScaled=x_scaler.transform(xUnscaled)
    print(xScaled.shape)
    print(xScaled[-5:])

    # # Way1
    # xScaledToPredict = []
    # xScaledToPredict.append(xScaled)
    # print(len(xScaledToPredict))

    # yPredScaled=x_model.predict(np.array(xScaledToPredict))
    # print(yPredScaled.shape,yPredScaled)

    # yPred  = x_scalerPred.inverse_transform(yPredScaled.reshape(-1, 1))
    # print(yPred.shape,yPred)

    #Way2
    xScaledToPredict= xScaled.reshape(1,input_sequence_length,len(feature_cols))
    print(xScaledToPredict.shape)

    yPredScaled = x_model.predict(xScaledToPredict)
    print(yPredScaled.shape, yPredScaled)

    yPred = x_scalerPred.inverse_transform(yPredScaled).reshape(-1, 1)
    print(yPred.shape, yPred)


    print("============================Summary============================")
    print(xUnscaled.shape)
    print(yPred.shape)

    # print("============================Input============================")
    # print(xUnscaled)
    # print("============================Output============================")
    # print(yPred)



    # # Build Predition Result Data

    # ## Feature Data

    # In[127]:

    print("Create indexes from Dataframe dfForPred")
    dfFeature=pd.DataFrame(data= xUnscaled,columns=feature_cols,index=dfForPred.index)

    print(dfFeature.shape)
    print(dfFeature.head())
    print(dfFeature.tail())


    # ## Forecast Value Data

    # In[128]:
    from pandas.tseries.holiday import USFederalHolidayCalendar
    from pandas.tseries.offsets import CustomBusinessDay
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

    print("Create indexes by specifying output_sequence_length stating from get last record of DFFeature+1")
    lastRowOfFeature=dfFeature.index.max()
    firstRowofPrediction=lastRowOfFeature+timedelta(days=1)
    datePred=pd.date_range(start=firstRowofPrediction,freq=us_bd,periods=output_sequence_length)
    print(datePred)

    dfPrediction=pd.DataFrame(data= yPred,columns=feature_cols,index=datePred)
    dfPrediction.index.name=date_col
    print(dfPrediction.shape)
    print(dfPrediction)


    # # Get Prepraed To ingest data into BQ , we have to create dataframe and convert to Json-Rowns

    # In[129]:


    outputDF=pd.DataFrame(data=[ [today,asset_name,prediction_col,dtStr_imported,model_id] ],columns=["prediction_date","asset_name","prediction_name","pred_timestamp","model_id"])
    print(outputDF.info())
    print(outputDF)


    # In[130]:


    jsonOutput = json.loads(outputDF.to_json(orient = 'records'))
    for item in jsonOutput:

        dataFeature=dfFeature.reset_index()[[date_col,prediction_col]]
        dataFeature[date_col]=dataFeature[date_col].dt.strftime('%Y-%m-%d')
        dataFeature.columns=["input_date","input_feature"]
        jsonFeature= json.loads(dataFeature.to_json(orient = 'records'))
        item["feature_for_prediction"]=jsonFeature

        dataPred=dfPrediction.reset_index()[[date_col,prediction_col]]
        dataPred[date_col]=dataPred[date_col].dt.strftime('%Y-%m-%d')
        dataPred.columns=["output_date","output_value"]
        jsonPred= json.loads(dataPred.to_json(orient = 'records'))
        item["prediction_result"]=jsonPred

    with open("fin_prediction.json", "w") as outfile:
        json.dump(jsonOutput, outfile)
    jsonOutput


    # # Ingest Data to BigQuery 

    # In[131]:


    try:
        table=client.get_table(table_id)
        print("Table {} already exists.".format(table_id))
        print(table.schema)
    except Exception as ex :
        print(str(ex))
    #if error  please create table and other configuration as  bq_prediction.txt    

    job_config = bigquery.LoadJobConfig(
    # schema=[  ]
    )

    job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND  
    #job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
    job = client.load_table_from_json(jsonOutput,table_id, job_config = job_config)
    if job.errors is not None:
        print(job.error_result)
        print(job.errors)
    else:
        print(f"Import to bigquery successfully  {len(jsonOutput)} records")

    #job_config.schema


    # In[132]:


    return   f"The system has done predicting price movement of {asset_name}-{prediction_col}-{today}."


    # In[ ]:





    # In[ ]:





    # In[ ]:




