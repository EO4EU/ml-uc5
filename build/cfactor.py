from flask import Flask, request, make_response
import uuid
import json
import kubernetes
from kubernetes import client, config
import sys
from cloudpathlib import CloudPath
from cloudpathlib import S3Client
import tritonclient.http.aio as httpclient
import numpy as np
if sys.version_info >= (3, 12, 0):
      import six
      sys.modules['kafka.vendor.six.moves'] = six.moves
from kafka import KafkaProducer
import os
import tarfile
import lzma
import traceback
import logging

from io import BytesIO

import pickle
import base64

import re

import subprocess

import tempfile

from pathlib import Path

import threading

import rasterio

import numpy as np

import asyncio
import csv

import time

import pandas as pd

import functools

from KafkaHandler import KafkaHandler,DefaultContextFilter

import joblib

import cv2

def create_app():

      app = Flask(__name__)
      app.logger.setLevel(logging.DEBUG)
      handler = KafkaHandler()
      handler.setLevel(logging.INFO)
      filter = DefaultContextFilter()
      app.logger.addHandler(handler)
      app.logger.addFilter(filter)
      app.logger.info("Application Starting up...", extra={'status': 'DEBUG'})
      scaler_params = joblib.load("scaler_params.joblib")
      target_mean = joblib.load("target_mean.joblib")

      def calculate_spectral_indices(df):
            """Calculate spectral indices from satellite bands"""
            df = df.copy()
            
            # NDVI (Normalized Difference Vegetation Index)
            df['NDVI'] = (df['B8'] - df['B4']) / (df['B8'] + df['B4'] + 1e-8)
            
            # NDWI (Normalized Difference Water Index)
            df['NDWI'] = (df['B3'] - df['B8']) / (df['B3'] + df['B8'] + 1e-8)
            
            # Simple Ratio (SR)
            df['SR'] = df['B8'] / (df['B4'] + 1e-8)
            
            # Enhanced Vegetation Index (EVI)
            df['EVI'] = 2.5 * ((df['B8'] - df['B4']) / (df['B8'] + 6 * df['B4'] - 7.5 * df['B2'] + 1))
            
            # Soil Adjusted Vegetation Index (SAVI)
            df['SAVI'] = ((df['B8'] - df['B4']) / (df['B8'] + df['B4'] + 0.5)) * 1.5
            
            return df

      def calculate_band_ratios(df):
            """Calculate ratios between different spectral bands"""
            df = df.copy()
            band_cols = ['B11', 'B12', 'B2', 'B3', 'B4', 'B8']
            
            # Calculate all combinations of band ratios
            for i, band1 in enumerate(band_cols):
                  for band2 in band_cols[i+1:]:
                        ratio_name = f'{band1}_{band2}_ratio'
                        df[ratio_name] = df[band1] / (df[band2] + 1e-8)
            
            return df

      def calculate_polynomial_features(df):
            """Calculate polynomial features for important spectral indices"""
            df = df.copy()
            important_features = ['NDVI', 'EVI', 'SAVI']
            
            for feature in important_features:
                  if feature in df.columns:
                        df[f'{feature}_pow2'] = df[feature] ** 2
            
            return df

      # This is the entry point for the SSL model from Image to Feature service.
      # It will receive a message from the Kafka topic and then do the inference on the data.
      # The result will be sent to the next service.
      # The message received should be a json with the following fields:
      # previous_component_end : A boolean that indicate if the previous component has finished.
      # S3_bucket_desc : A json with the following fields:
      # folder : The folder where the data is stored.
      # The namespace of the configmap to read is the name of the pod.
      # The name of the configmap to read is given by the URL.
      # The configmap should have a field named jsonSuperviserRequest that is a json with the following fields:
      # Topics : A json with the following fields:
      # out : The name of the kafka topic to send the result.
      # S3_bucket : A json with the following fields:
      # aws_access_key_id : The access key id of the S3 bucket.
      # aws_secret_access_key : The secret access key of the S3 bucket.
      # s3-bucket_name : The name of the S3 bucket.
      # region_name : The name of the region of the S3 bucket.
      # endpoint_url : The endpoint url of the S3 bucket.
      # ML : A json with the following fields:
      # need-to-resize : A boolean that indicate if the data need to be resized.

      @app.route('/<name>', methods=['POST'])
      def cfactor(name):
            time_start = time.time()
            response=None
            # TODO : Debugging message to remove in production.
            # Message received.
            try:
                  raw_data = request.data
                  def threadentry(raw_data):
                        app.logger.info('Received message for '+str(name))
                        app.logger.info(f"thread ID is {threading.get_ident()}")
                        config.load_incluster_config()
                        api_instance = client.CoreV1Api()
                        configmap_name = str(name)
                        configmap_namespace = 'uc5'
                        api_response = api_instance.read_namespaced_config_map(configmap_name, configmap_namespace)
                        json_data_request = json.loads(raw_data)
                        json_data_configmap =json.loads(str(api_response.data['jsonSuperviserRequest']))
                        workflow_name = json_data_configmap.get('workflow_name', '')
                        bootstrapServers =api_response.data['bootstrapServers']
                        component_name = json_data_configmap['ML']['component_name']
                        

                        kafka_out = json_data_configmap['Topics']["out"]
                        s3_access_key = json_data_configmap['S3_bucket']['aws_access_key_id']
                        s3_secret_key = json_data_configmap['S3_bucket']['aws_secret_access_key']
                        s3_bucket_output = json_data_configmap['S3_bucket']['s3-bucket-name']
                        s3_region = json_data_configmap['S3_bucket']['region_name']
                        s3_region_endpoint = json_data_configmap['S3_bucket']['endpoint_url']

                        s3_path = json_data_request['S3_bucket_desc']['folder']
                        #s3_file = json_data_request['S3_bucket_desc'].get('filename',None)
                        while True:
                              try:
                                    Producer=KafkaProducer(bootstrap_servers=bootstrapServers,value_serializer=lambda v: json.dumps(v).encode('utf-8'),key_serializer=str.encode)
                                    break
                              except Exception as e:
                                    app.logger.error('Got exception '+str(e)+'\n'+traceback.format_exc()+'\n'+'So we retry', extra={'status': 'CRITICAL'})
                        try:
                              logger_workflow = logging.LoggerAdapter(app.logger, {'source': component_name,'workflow_name': workflow_name,'producer':Producer},merge_extra=True)
                              logger_workflow.info('Starting Workflow',extra={'status':'START'})
                              logger_workflow.debug('Reading json data request'+str(json_data_request), extra={'status': 'DEBUG'})
                              logger_workflow.debug('Reading json data configmap'+str(json_data_configmap), extra={'status': 'DEBUG'})
                              if not(json_data_request['previous_component_end'] == 'True' or json_data_request['previous_component_end']):
                                    class PreviousComponentEndException(Exception):
                                          pass
                                    raise PreviousComponentEndException('Previous component did not end correctly')

                              logger_workflow.debug('All json data read', extra={'status': 'DEBUG'})

                              clientS3 = S3Client(aws_access_key_id=s3_access_key, aws_secret_access_key=s3_secret_key,endpoint_url=s3_region_endpoint)
                              clientS3.set_as_default_client()

                              logger_workflow.debug('Client is ready', extra={'status': 'DEBUG'})
                              if s3_path.endswith('/'):
                                    s3_path=s3_path[:-1]
                              cp = CloudPath("s3://"+s3_bucket_output+'/'+s3_path+'/', client=clientS3)
                              cpOutput = CloudPath("s3://"+s3_bucket_output+'/result-uc5-cfactor/')
                              logger_workflow.debug("path is s3://"+s3_bucket_output+'/result-uc5-cfactor/', extra={'status': 'DEBUG'})

                              with cpOutput.joinpath('log.txt').open('w') as fileOutput:
                                    meta=None
                                    def treatFolder(folder):
                                                pattern=r'.*MSIL2A.*\.SAFE$'
                                                match = re.search(pattern,folder.name)
                                                if match:
                                                      logger_workflow.debug('matched folder '+str(folder), extra={'status': 'DEBUG'})
                                                      with tempfile.TemporaryDirectory() as tempdirBen:
                                                            cpGranule=folder/"GRANULE"
                                                            dicPath={}
                                                            for folderGranule in cpGranule.iterdir():
                                                                  logger_workflow.debug("granule path "+str(folderGranule),extra={'status': 'DEBUG'})
                                                                  cpIMGData10m=folderGranule/"IMG_DATA"/"R10m"
                                                                  for image in cpIMGData10m.iterdir():
                                                                        pattern=r'.*_(.*)_10m\.jp2$'
                                                                        match = re.search(pattern,image.name)
                                                                        logger_workflow.debug("image path "+str(image),extra={'status': 'DEBUG'})
                                                                        if match:
                                                                              matchedBand=match.group(1)
                                                                              logger_workflow.debug("matchedBand "+matchedBand,extra={'status': 'DEBUG'})
                                                                              if matchedBand in ['B02','B03','B04','B08']:
                                                                                    logger_workflow.debug("matchedBand 10m "+matchedBand,extra={'status': 'DEBUG'})
                                                                                    path_src=image
                                                                                    dicPath[matchedBand]=path_src
                                                                        else:
                                                                              logger_workflow.debug("not matched",extra={'status': 'DEBUG'})
                                                                  cpIMGData20m=folderGranule/"IMG_DATA"/"R20m"
                                                                  for image in cpIMGData20m.iterdir():
                                                                        pattern=r'.*_(.*)_20m\.jp2$'
                                                                        match = re.search(pattern,image.name)
                                                                        logger_workflow.debug("image path "+str(image),extra={'status': 'DEBUG'})
                                                                        if match:
                                                                              logger_workflow.debug("matched",extra={'status': 'DEBUG'})
                                                                              matchedBand=match.group(1)
                                                                              logger_workflow.debug("matchedBand "+matchedBand,extra={'status': 'DEBUG'})
                                                                              if matchedBand in ['B05','B06','B07','B8A','B11','B12']:
                                                                                    logger_workflow.debug("matchedBand 20m "+matchedBand,extra={'status': 'DEBUG'})
                                                                                    path_src=image
                                                                                    dicPath[matchedBand]=path_src
                                                                        else:
                                                                              logger_workflow.debug("not matched",extra={'status': 'DEBUG'})
                                                                  BANDS_10M = [
                                                                              "B04",
                                                                              "B03",
                                                                              "B02",
                                                                              "B08",
                                                                              ]


                                                                  BANDS_20M = [
                                                                  "B11",
                                                                  "B12",
                                                                  ]
                                                                  
                                                                  BANDS_ALL=BANDS_10M+BANDS_20M
                                                                  bands_data = {}
                                                                  metaData={}
                                                                  for band_name in BANDS_ALL:
                                                                        if band_name not in dicPath:
                                                                              logger_workflow.debug("band_name "+band_name+" not found. Stopping treating folder "+str(folder),extra={'status': 'INFO'})
                                                                              return
                                                                        band_path = dicPath[band_name]
                                                                        logger_workflow.debug("band_path "+str(band_path),extra={'status': 'DEBUG'})
                                                                        with band_path.open('rb') as fileBand, rasterio.io.MemoryFile(fileBand) as memfile:
                                                                              with memfile.open(sharing=False) as band_file:
                                                                                    band_data   = band_file.read(1,masked=True)  # open the tif image as a numpy array
                                                                                    band_data=band_data.filled(np.nan)
                                                                                    metaData[band_name] = band_file.meta
                                                                                    # Resize depending on the resolution
                                                                                    if band_name in BANDS_20M:
                                                                                          h=band_data.shape[0]
                                                                                          w=band_data.shape[1]
                                                                                          # Carry out a bicubic interpolation (TUB does exactly this)
                                                                                          band_data = cv2.resize(band_data, dsize=(2*h, 2*w), interpolation=cv2.INTER_CUBIC)
                                                                                          # We have already ignored the 60M ones, and we keep the 10M ones intact
                                                                                    #logging.info("appending")
                                                                                    bands_data[band_name] = band_data
                                                                                    logger_workflow.debug("band_name "+band_name,extra={'status': 'DEBUG'})
                                                                                    logger_workflow.debug("band_data shape "+str(band_data.shape),extra={'status': 'DEBUG'})
                                                                        band_file.close()
                                                                  value=pd.DataFrame()
                                                                  h=None
                                                                  w=None
                                                                  for band_name in BANDS_ALL:
                                                                        band_data=bands_data[band_name]
                                                                        h_band=band_data.shape[0]
                                                                        w_band=band_data.shape[1]
                                                                        if h is None:
                                                                              h=h_band
                                                                        if w is None:
                                                                              w=w_band
                                                                        if h!=h_band or w!=w_band:
                                                                              logger_workflow.debug("Different shape found for band "+band_name+" h "+str(h)+" w "+str(w)+" h_band "+str(h_band)+" w_band "+str(w_band)+" Stopping treating folder "+str(folder),extra={'status': 'INFO'})
                                                                              return
                                                                  for i in range(0,h):
                                                                        for j in range(0,w):
                                                                              dic={}
                                                                              for band_name in BANDS_ALL:
                                                                                    band_data=bands_data[band_name]
                                                                                    dic[band_name]=band_data[i,j]
                                                                              dic['x']=i
                                                                              dic['y']=j
                                                                              value=value.append(dic,ignore_index=True)
                                                                  logger_workflow.debug(f"Processing {len(value)} samples...", extra={'status': 'DEBUG'})
                                                                  value = calculate_spectral_indices(value)
                                                                  value = calculate_band_ratios(value)
                                                                  value = calculate_polynomial_features(value)

                                                                  # Feature order (must match training)
                                                                  feature_names = [
                                                                  # Original bands
                                                                  'B11', 'B12', 'B2', 'B3', 'B4', 'B8',
                                                                  # Spectral indices
                                                                  'NDVI', 'NDWI', 'SR', 'EVI', 'SAVI',
                                                                  # Band ratios (15 combinations for 6 bands)
                                                                  'B11_B12_ratio', 'B11_B2_ratio', 'B11_B3_ratio', 'B11_B4_ratio', 'B11_B8_ratio',
                                                                  'B12_B2_ratio', 'B12_B3_ratio', 'B12_B4_ratio', 'B12_B8_ratio',
                                                                  'B2_B3_ratio', 'B2_B4_ratio', 'B2_B8_ratio',
                                                                  'B3_B4_ratio', 'B3_B8_ratio',
                                                                  'B4_B8_ratio',
                                                                  # Polynomial features
                                                                  'NDVI_pow2', 'EVI_pow2', 'SAVI_pow2'
                                                                  ]

                                                                  # Extract features in correct order
                                                                  X = value[feature_names].values
                                                                  print(f"Feature matrix shape: {X.shape}")

                                                                  # === Apply robust scaling ===
                                                                  # During training, all features were scaled using robust scaling (no time features):
                                                                  # scaled = (x - q1) / IQR
                                                                  X_scaled = X.copy()
                                                                  for i in range(X.shape[1]):  # Scale all features
                                                                        fname = list(scaler_params['iqr'].keys())[i]
                                                                        q1 = scaler_params['q1'][fname]
                                                                        iqr = scaler_params['iqr'][fname]
                                                                        X_scaled[:, i] = (X[:, i] - q1) / iqr
                                                                  toInfer = []
                                                                  for i in range(0,X_scaled.shape[0]):
                                                                        dic={}
                                                                        dic["i"]=i
                                                                        dic["data"]=X_scaled[i:i+1,:].astype(np.float32)
                                                                        toInfer.append(dic)
                                                                  logger_workflow.debug('start inference', extra={'status': 'DEBUG'})
                                                                  logger_workflow.debug('length '+str(len(toInfer)), extra={'status': 'DEBUG'})
                                                                  asyncio.run(doInference(toInfer,logger_workflow))
                                                                  logger_workflow.debug('inference done', extra={'status': 'DEBUG'})
                                                                  resultArray=np.zeros((X_scaled.shape[0],3),dtype=np.float32)
                                                                  for requestElem in toInfer:
                                                                        result_subarray=requestElem["result"]
                                                                        i=requestElem["i"]
                                                                        resultArray[i,0]=result_subarray + target_mean - 0.5
                                                                        resultArray[i,1]=requestElem["x"]
                                                                        resultArray[i,2]=requestElem["y"]
                                                                  logger_workflow.debug('array all done', extra={'status': 'DEBUG'})

                                                                  df_result = pd.DataFrame(resultArray, columns=['cfactor','x','y'])
                                                                  outputPath=cpOutput.joinpath(key+'-cfactor-result.csv')
                                                                  logger_workflow.debug('csv writting', extra={'status': 'DEBUG'})
                                                                  with outputPath.open('w') as outputFile:
                                                                        df_result.to_csv(outputFile, index=False,header=True)
                                                                  logger_workflow.debug('csv writting done', extra={'status': 'DEBUG'})

                                                                  outputPath=cpOutput.joinpath(key+'-cfactor-result.jp2')

                                                                  array=np.zeros((h,w),dtype=np.float32)
                                                                  for i in range(0,w):
                                                                        for j in range(0,h):
                                                                              array[j,i]=resultArray[j*w+i,0]

                                                                  with outputPath.open('wb') as outputFile,rasterio.io.MemoryFile() as memfile:
                                                                        #with rasterio.open(outputFile,mode='w',**data["meta"][ALL_BANDS[band_number]]) as file2:
                                                                        with memfile.open(driver="JP2OpenJPEG",width=w,height=h,count=1,dtype="fp32",crs=metaData["B3"]["crs"],transform=metaData["B3"]["transform"]) as file2:
                                                                              file2.write(array, indexes=1)
                                                                        outputFile.write(memfile.read())
                                    def recurse_folders(cp):
                                          for folder in cp.iterdir():
                                                treatFolder(folder)
                                                recurse_folders(folder)

                                    recurse_folders(cp)
                                    
                                    logger_workflow.debug('Output written', extra={'status': 'DEBUG'})
                                    logger_workflow.debug('Connecting to Kafka', extra={'status': 'DEBUG'})
            
                                    response_json ={
                                    "previous_component_end": "True",
                                    "S3_bucket_desc": {
                                          "folder": "result-uc6-classifier","filename": ""
                                    },
                                    "meta_information": json_data_request.get('meta_information',{})}
                                    Producer.send(kafka_out,key='key',value=response_json)
                                    Producer.flush()
                        except Exception as e:
                              logger_workflow.error('Got exception '+str(e)+'\n'+traceback.format_exc()+'\n'+'So we are ignoring the message', extra={'status': 'CRITICAL'})
                              return
                        logger_workflow.info('workflow finished successfully',extra={'status':'SUCCESS'})

                  thread = threading.Thread(target=threadentry, args=(raw_data,))
                  thread.start()
                  app.logger.info('total time '+str(time.time()-time_start))
                  app.logger.info('Thread started')
                  response = make_response({
                              "msg": "Started the process"
                              })
                  app.logger.info('made response')
            except Exception as e:
                  app.logger.error('Got exception '+str(e)+'\n'+traceback.format_exc()+'\n'+'So we are ignoring the message', extra={'status': 'CRITICAL'})
                  # HTTP answer that the message is malformed. This message will then be discarded only the fact that a sucess return code is returned is important.
                  response = make_response({
                  "msg": "There was a problem ignoring"
                  },500)
            return response

      # This function is used to do the inference on the data.
      # It will connect to the triton server and send the data to it.
      # The result will be returned.
      # The data should be a numpy array of shape (1,10,120,120) and type float32.
      # The result will be a json with the following fields:
      # model_name : The name of the model used.
      # outputs : The result of the inference.
      async def doInference(toInfer,logger_workflow):

            triton_client = httpclient.InferenceServerClient(url="default-inference.uc5.svc.cineca-inference-server.local", verbose=False,conn_timeout=10000000,conn_limit=None,ssl=False)
            nb_Created=0
            nb_InferenceDone=0
            nb_Postprocess=0
            nb_done_instance=0
            list_postprocess=set()
            list_task=set()
            last_throw=0
            async def consume(task):
                  try:
                        if task[0]==1:
                              count=task[1]
                              inputs=[]
                              outputs=[]
                              inputs.append(httpclient.InferInput('input__0',toInfer[count]["data"].shape, "FP32"))
                              inputs[0].set_data_from_numpy(toInfer[count]["data"], binary_data=True)
                              outputs.append(httpclient.InferRequestedOutput('output__0', binary_data=True))
                              results = await triton_client.infer('cfactor2',inputs,outputs=outputs)
                              return (task,results)
                        if task[0]==255:
                              count=task[1]
                              inputs = []
                              outputs = []
                              input=np.zeros([255,toInfer[count]["data"].shape[1]],dtype=np.float32)
                              for i in range(0,255):
                                    input[i]=toInfer[count+i]["data"][0]
                              inputs.append(httpclient.InferInput('input__0',input.shape, "FP32"))
                              inputs[0].set_data_from_numpy(input, binary_data=True)
                              outputs.append(httpclient.InferRequestedOutput('output__0', binary_data=True))
                              results = await triton_client.infer('cfactor2',inputs,outputs=outputs)
                              return (task,results)
                  except Exception as e:
                        logger_workflow.debug('Got exception in inference '+str(e)+'\n'+traceback.format_exc(), extra={'status': 'WARNING'})
                        nonlocal last_throw
                        last_throw=time.time()
                        return await consume(task)
            
            async def postprocess(task,results):
                  if task[0]==1:
                        result=results.as_numpy('output__0')[0]
                        toInfer[task[1]]["result"]=result
                  if task[0]==255:
                        result=results.as_numpy('output__0')
                        for i in range(0,255):
                              toInfer[task[1]+i]["result"]=result[i]
                        

            def postprocessTask(task):
                  list_task.discard(task)
                  new_task=asyncio.create_task(postprocess(*task.result()))
                  list_postprocess.add(new_task)
                  def postprocessTaskDone(task2):
                        nonlocal nb_Postprocess
                        nb_Postprocess+=1
                        nonlocal nb_done_instance
                        nb_done_instance+=task.result()[0][0]
                        list_postprocess.discard(task2)
                  new_task.add_done_callback(postprocessTaskDone)
                  nonlocal nb_InferenceDone
                  nb_InferenceDone+=1


            def producer():
                  total=len(toInfer)
                  count=0
                  while total-count>=255:
                        yield (255,count)
                        count=count+255
                  while total-count>=1:
                        yield (1,count)
                        count=count+1
            
            last_shown=time.time()
            start=time.time()-60
            for item in producer():
                  while time.time()-last_throw<30 or nb_Created-nb_InferenceDone>(time.time()-start)*5 or nb_Postprocess-nb_InferenceDone>(time.time()-start)*5:
                        await asyncio.sleep(0)
                  task=asyncio.create_task(consume(item))
                  list_task.add(task)
                  task.add_done_callback(postprocessTask)
                  nb_Created+=1
                  if time.time()-last_shown>60:
                        last_shown=time.time()
                        logger_workflow.info('done instance '+str(nb_done_instance)+'Inference done value '+str(nb_InferenceDone)+' postprocess done '+str(nb_Postprocess)+ ' created '+str(nb_Created), extra={'status': 'DEBUG'})
            while nb_InferenceDone-nb_Created>0 or nb_Postprocess-nb_InferenceDone>0:
                  await asyncio.sleep(0)
            await asyncio.gather(*list_task,*list_postprocess)
            logger_workflow.info('Inference done',extra={'status':'DEBUG'})
            await triton_client.close()
      return app
      