#!/usr/bin/env python
# coding: utf-8

# In[130]:


import pandas as pd
import numpy as np
import os
from datetime import datetime,date,timedelta,timezone
import calendar
import json


from google.cloud import bigquery
from google.oauth2 import service_account
from google.cloud.exceptions import NotFound
from google.api_core.exceptions import BadRequest



# In[131]:



# uncomment and indent
import functions_framework
@functions_framework.http
def collect_prediction_result(request):   # run on clound function

# def collect_prediction_result(collectionDate): # backfill


# # Dict To store all collective data Mode To run Job

# In[132]:


    dictCollectPerf={}

    # uncomment
    mode=2 # 2 for prodictoin 1 for test/migrate 
    modelList=['spy-ema1-60t10-ds0115t0523','qqq-ema1-30t5-ds0115t0523','spy-signal-60t10-ds0115t0523']

    # comment
    #model_id='spy-ema1-60t10-ds0115t0523'
    # model_id='qqq-ema1-30t5-ds0115t0523'
    #model_id="spy-signal-60t10-ds0115t0523"


    # # Init parameter

    # In[133]:


    if mode==1: # Migrate to backfill data and Test 
        logDate=collectionDate
        log_date=datetime.strptime(logDate,'%Y-%m-%d %H:%M')
        log_timestamp=datetime.strptime(logDate,'%Y-%m-%d %H:%M')
    else: # On weekly basis
        log_timestamp=datetime.now(timezone.utc)
        log_date=datetime.strptime(log_timestamp.strftime('%Y-%m-%d'),'%Y-%m-%d')

    week_day=log_date.weekday()
    day_name=calendar.day_name[log_date.weekday()]

    print(f"Date to collect data on {log_date.strftime('%Y-%m-%d')} {day_name}(Idx:{week_day}) at {log_timestamp}")
    #week_day=log_date.weekday() Sature=5  [0,1,2,3,4,5,6]

    # Friday of last n week so we want to collect model perf  0=last week ,1=2 week  
    # 2=3 is weeks it convert the Friday (last trading day) to have actual value at least  2 week(10day) ahead to compare  predicted value
    no_week_lookback=2
    if  week_day==5:
        last_trading_day_of_week=1+(no_week_lookback*7) # 7 is 1 week
        # last_trading_day_of_week=1
    else:
        # comment
        # raise Exception("Saturday is allowed  as Collection Date for forcasting result.")   
        # uncomment
        return "Saturday is allowed  as Collection Date for forcasting result."  

    print(f"week_day={week_day} and last_trading_day_of_week={last_trading_day_of_week}")

    genTableSchema=False
    metric_name='mae'


    # # Create Start to End Date By Getting Last Date of Week

    # In[134]:


    # get  prev prediction  from  get end prediction to beginneg or predicton of week 
    endX=log_date+timedelta(days=-last_trading_day_of_week)
    startX=endX+timedelta(days=1)+timedelta(days=-5) #-5 is from Friday to Monday
    print(f"Collection data from {startX.strftime('%A %d-%m-%Y')} to {endX.strftime('%A %d-%m-%Y')}")

    endX=endX.strftime('%Y-%m-%d')
    startX=startX.strftime('%Y-%m-%d')

    print(f"Convert Start and End Date to gather data {startX} - {endX} to string")


    # # BigQuery Setting & Configuration Variable

    # In[108]:


    date_col='date'
    projectId='pongthorn'
    dataset_id='FinAssetForecast'

    table_data_id=f"{projectId}.{dataset_id}.fin_data"
    table_id = f"{projectId}.{dataset_id}.fin_movement_forecast"
    table_model_id= f"{projectId}.{dataset_id}.model_ts_metadata"
    
    table_perf_id= f"{projectId}.{dataset_id}.model_forecast_performance"
    #table_perf_id= f"{projectId}.{dataset_id}.model2_demo_forecast_performance"

    print(table_id)
    print(table_data_id)
    print(table_model_id)
    print(table_perf_id)

    client = bigquery.Client(project=projectId )

    def load_data_bq(sql:str):
        query_result=client.query(sql)
        df=query_result.to_dataframe()
        return df


    # In[ ]:





    # # Start Loop

    # In[109]:


    # uncomment  and indent
    def process_data(model_id):
        print(f"Collect data : {model_id} ")


        # # Check where the given date collected data or not?

        # In[110]:


        # this version , it will check on each model invidually
        # you can move it inside def process

        # script in the first run because we want to generate table with nested and repeated column
        sqlCheck=f"""
        select collection_timestamp from `{table_perf_id}`
        where date(collection_timestamp)='{log_date.strftime('%Y-%m-%d')}' and model_id='{model_id}'
        """

        print(sqlCheck)
        dfCheckDate=load_data_bq(sqlCheck)
        if  dfCheckDate.empty==False:
            print(f"Collection data on {log_date} for {model_id} found, no any action")
            # uncomment
            return f"Collection data on {log_date} for {model_id} found, no any action"
        else:
            print(f"We are ready to Collect data on {log_date}")


        # # Get Model Meta

        # In[111]:


        def get_model_metadata(model_id):
            sqlModelMt=f"""
            SELECT * FROM `{table_model_id}`  where model_id='{model_id}'
            """
            print(sqlModelMt)
            dfModelMeta=load_data_bq(sqlModelMt)
            return  dfModelMeta

        dfModelMeta=get_model_metadata(model_id)

        if dfModelMeta.empty==False:
            modelMeta=dfModelMeta.iloc[0,:]
            print(modelMeta)
            asset_name=modelMeta['asset']
            prediction=modelMeta['prediction']
        else: 
            raise Exception(f"Not found model id  {model_id}")


        # # Retrive forecasting result data to Dictionary

        # In[112]:


        def get_forecasting_result_data(request):

            if   request is not None:  
                start_date=request["start_date"]
                end_date=request["end_date"]
                prediction_name=request["prediction_name"]
                asset_name=request["asset_name"]
                model_id=request["model_id"]
            else:
                raise Exception("No request parameters such as start_date,prediction_name,asset_name")

            
            print("1.How far in advance does model want to  make prediction")
            sqlOutput=f"""
            select t.prediction_date, t.pred_timestamp,t.asset_name,t.prediction_name,
            t_pred.output_date as {date_col},t_pred.output_value as {prediction_name}
            from  `{table_id}` t
            cross join unnest(t.prediction_result) t_pred
            where (t.prediction_date>='{start_date}' and  t.prediction_date<='{end_date}')
            and t.model_id='{model_id}'
            order by  t.prediction_date,t_pred.output_date
            """
            print(sqlOutput)
            dfOutput=load_data_bq(sqlOutput)
            # dfOutput=dfOutput.drop_duplicates(subset=[date_col,'asset_name','prediction_name'],keep='last',)
            # dfOutput=dfOutput.drop_duplicates(subset=[date_col],keep='last',)
            dfOutput[date_col]=pd.to_datetime(dfOutput[date_col],format='%Y-%m-%d')
            dfOutput.set_index(date_col,inplace=True)

            output_sequence_length=len(dfOutput)
            print(f"output_sequence_length={output_sequence_length}")
            

            print(dfOutput.info())
            print(dfOutput[['prediction_date','asset_name','prediction_name' ,prediction_name]])
            print("================================================================================================")

            
            #get actual data since the fist day of input and the last day of output(if covered)
            startFinData=dfOutput.index.min().strftime('%Y-%m-%d')
            endFindData=dfOutput.index.max().strftime('%Y-%m-%d')
            print(f"2.Get Real Data  to compare to prediction from {startFinData} to {endFindData}")

            sqlData=f"""
            select Date as {date_col},{prediction_name}, ImportDateTime, from `{table_data_id}` 
            where (Date>='{startFinData}' and Date<='{endFindData}') and Symbol='{asset_name}'
            order by ImportDateTime,Date
            """
            print(sqlData)

            dfRealData=load_data_bq(sqlData)
            dfRealData=dfRealData.drop_duplicates(subset=[date_col],keep='last',)
            dfRealData[date_col]=pd.to_datetime(dfRealData[date_col],format='%Y-%m-%d')
            dfRealData.set_index(date_col,inplace=True)
            
            print(dfRealData.info())
            print(dfRealData[[prediction_name]])
            print("================================================================================================")

            return {'actual_price':dfRealData,'output':dfOutput }


        print(f"================Get data from {startX}====to==={endX}================")
        request={'start_date':startX,'end_date':endX,'prediction_name':prediction,'asset_name':asset_name,'model_id':model_id}
        data=get_forecasting_result_data(request)
        print(f"=======================================================================")


        # # Create Predictive and Actual Value dataframe

        # In[113]:


        print("List all trading day in the week")
        myTradingDataList=data['output']['prediction_date'].unique()
        print(myTradingDataList)


        # In[114]:


        dfAllForecastResult=pd.DataFrame(columns=['date','pred_value','actual_value','prediction_date'])
        dfAllForecastResult


        # In[115]:


        print(f"========================dfX :Actual Price========================")
        dfX=data['actual_price'][[prediction]]
        dfX.columns=[f'actual_value']
        print(dfX.info())
        dfX


        # In[116]:


        # actually , we can jon without spilting data by prediction_dtate
        for date in  myTradingDataList: # trading day on giver week
            print(f"=========================dfPred:Predicted Price at {date}=========================")
            dfPred=data['output'].query("prediction_date==@date")[[prediction]]
            dfPred.columns=[f'pred_value']
            print(dfPred)
            print(dfPred.info())

            print("=====================dfCompare:Join Actual price to Predicted Price=================")
            dfCompare=pd.merge(left=dfPred,right=dfX,how='inner',right_index=True,left_index=True)
            dfCompare.reset_index(inplace=True)   
            dfCompare['prediction_date']=date.strftime('%Y-%m-%d')      
            print(dfCompare) 
            print(dfCompare.info())

            if len(dfCompare)>0 : # it will be join if there is at least one record to show actual vs pred
                dfAllForecastResult= pd.concat([dfAllForecastResult,dfCompare],ignore_index=True)
                print(f"=========================Appended Data Joined=========================")
            else:
                print("No Appendind Data due to no at least one record to show actual vs pred")  
            


        # In[117]:


        print("========================dfAllForecastResult: All Predicton Result========================")
        print(dfAllForecastResult.info())
        print(dfAllForecastResult)


        # # Calculate MAE Metric

        # ## Get sum distance between pred and actul value from prev rows

        # In[118]:


        sqlMetric=f"""
        with pred_actual_by_model as  
        (
        SELECT  detail.actual_value,detail.pred_value
        from `{table_perf_id}`  t
        cross join unnest(t.pred_actual_data) as detail
        where t.model_id='{model_id}' and t.collection_timestamp<'{log_timestamp}'
        )
        select COALESCE( sum(abs(x.actual_value-x.pred_value)),0) as pred_diff_actual,count(*) as no_row  from pred_actual_by_model  x


        """

        if genTableSchema==False:
            print(sqlMetric)

            dfMetric=load_data_bq(sqlMetric)
            prevSum=dfMetric.iloc[0,0]
            prevCount=dfMetric.iloc[0,1]

        else:  # it is used if there are something changed in table schema
        # for generating table schema
            prevSum=0
            prevCount=0

        print(f"Prev Sum={prevSum} and Count={prevCount}")


        # ## Cal sum distance between pred and actul value from last rows

        # In[119]:


        dfAllForecastResult['pred_diff_actual']=dfAllForecastResult.apply(lambda x : abs(x['pred_value']-x['actual_value']),axis=1)
        dfAllForecastResult


        # In[120]:


        recentSum=dfAllForecastResult['pred_diff_actual'].sum()
        recentCount=len(dfAllForecastResult)


        dfAllForecastResult=dfAllForecastResult.drop(columns=['pred_diff_actual'])
        print(f"Recent Sum={recentSum} and Count={recentCount}")


        # ## Calculate MEA as formula

        # In[121]:


        #https://en.wikipedia.org/wiki/Mean_absolute_error
        metric_value= round((prevSum+recentSum)/(prevCount+recentCount),2)
        print(f"{metric_name} = {metric_value}")


        # # Create Collection Performance Info Dataframe and Store 
        # 

        # In[122]:


        df=pd.DataFrame(data=[ [log_date,model_id,metric_name,metric_value,log_timestamp] ],
                        columns=["collection_date","model_id","metric_name","metric_value","collection_timestamp"])
        print(df.info())
        print(df)


        # In[123]:


        dictCollectPerf[model_id]=(df,dfAllForecastResult)


        # In[124]:


        # uncomment
        return f"gather data of {model_id}"


    # # End Loop

    # In[125]:


    # Iterate over model list
    # uncomment
    for modelID in modelList:
    # indent
      print(process_data(modelID))
      print("#########################################################")


    # # Create Json Data 

    # In[126]:


    jsonDataList=[]
    for model_id,dataTuple in  dictCollectPerf.items():
        print(model_id)
        
        masterDF=dataTuple[0]
        masterDF["collection_date"]=masterDF["collection_date"].dt.strftime('%Y-%m-%d')
        masterDF["collection_timestamp"]=masterDF["collection_timestamp"].dt.strftime('%Y-%m-%d %H:%M:%S')
        master_perf = json.loads(masterDF.to_json(orient = 'records'))[0] # 1 main dataframe has 1 records
        
        detailDF=dataTuple[1]  
        # print(detailDF.info())
        
        #detailDF["prediction_date"]=detailDF["prediction_date"].dt.strftime('%Y-%m-%d')
        detailDF["date"]=detailDF["date"].dt.strftime('%Y-%m-%d')
        
        detail_perf= json.loads(detailDF.to_json(orient = 'records'))
        master_perf["pred_actual_data"]=detail_perf
        
        jsonDataList.append(master_perf)
        
    # with open("fin_forecast_performance.json", "w") as outfile:
    #     json.dump( jsonDataList, outfile)


    # In[ ]:





    # # Ingest Data to BigQuery

    # ## Try to ingest data to get correct schema and copy the schema to create table including partion/cluster manually

    # In[127]:


    try:
        table=client.get_table(table_perf_id)
        print("Table {} already exists.".format(table_perf_id))
        # print(table.schema)
    except Exception as ex :
        print(str(ex))


    # In[128]:


    job_config = bigquery.LoadJobConfig()

    job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON

    # Try to ingest data to get correct schema and copy the schema to create table including partiion/cluster manually
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND 


    job = client.load_table_from_json(jsonDataList,table_perf_id, job_config = job_config)
    if job.errors is not None:
        print(job.error_result)
        print(job.errors)
        # uncomment
        # return "Error to load data to BigQuery"
    else:
        print(f"Import to bigquery successfully  {len(jsonDataList)} records")
        
    #job_config.schema
    # truncate table`pongthorn.FinAssetForecast.model_forecast_performance` 


    # In[129]:


    # uncomment
    return 'completely'


# In[ ]:





# In[ ]:


# uncomment
# Main 
# print("Collect prediction result to monitor performance model")
# start_backfill='2023-06-03 10:00' # comment
# end_backfill='2023-08-26 10:00'
# period_index=pd.date_range(start=start_backfill,end=end_backfill, freq="W-SAT")
# listLogDate=[ d.strftime('%Y-%m-%d %H:%M')   for  d in  period_index   ]
# for d in listLogDate:
#     print(d)
# multiple items

# listLogDate=[
#      '2023-08-05 00:00','2023-08-12 00:00','2023-08-19 00:00','2023-08-26 00:00','2023-09-02 00:00'
# ]
# for  d in listLogDate:
#   print(f"*******************************Collect prediction result as of {d}*****************************************")
#   print(collect_prediction_result(d))
#   print("************************************************************************************************")

# sigle item
# collectionDate='2023-09-09 10:00' # comment    
# print(collect_prediction_result(collectionDate))


# In[ ]:




