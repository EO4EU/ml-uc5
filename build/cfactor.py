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

def create_app():

      app = Flask(__name__)

      Producer=KafkaProducer(bootstrap_servers="kafka-external.dev.apps.eo4eu.eu:9092",value_serializer=lambda v: json.dumps(v).encode('utf-8'),key_serializer=str.encode)
      handler = KafkaHandler(producer=Producer,source='ML.uc5.cfactor')
      console_handler = logging.StreamHandler()
      console_handler.setLevel(logging.DEBUG)
      filter = DefaultContextFilter()
      app.logger.addFilter(filter)
      app.logger.addHandler(handler)
      app.logger.addHandler(console_handler)
      app.logger.setLevel(logging.DEBUG)

      app.logger.info("Starting up...", extra={'logName': 'startup'})

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
            app.logger.info('received request', extra={'logName': 'request'})
            # TODO : Debugging message to remove in production.
            # Message received.
            response=None
            try:
                  config.load_incluster_config()
                  api_instance = client.CoreV1Api()
                  configmap_name = str(name)
                  configmap_namespace = 'uc5'
                  app.logger.info('Namespace '+str(configmap_namespace), extra={'logName': 'namespace'})
                  api_response = api_instance.read_namespaced_config_map(configmap_name, configmap_namespace)
                  json_data_request = json.loads(request.data)
                  json_data_configmap =json.loads(str(api_response.data['jsonSuperviserRequest']))
                  bootstrapServers =api_response.data['bootstrapServers']
                  Producer=KafkaProducer(bootstrap_servers=bootstrapServers,value_serializer=lambda v: json.dumps(v).encode('utf-8'),key_serializer=str.encode)
                  app.logger.info('Reading json data request'+str(json_data_request), extra={'logName': 'json_data_request'})
                  app.logger.info('Reading json data configmap'+str(json_data_configmap), extra={'logName': 'json_data_configmap'})
                  assert json_data_request['previous_component_end'] == 'True' or json_data_request['previous_component_end']
                  kafka_out = json_data_configmap['Topics']["out"]
                  s3_access_key = json_data_configmap['S3_bucket']['aws_access_key_id']
                  s3_secret_key = json_data_configmap['S3_bucket']['aws_secret_access_key']
                  s3_bucket_output = json_data_configmap['S3_bucket']['s3-bucket-name']
                  s3_region = json_data_configmap['S3_bucket']['region_name']
                  s3_region_endpoint = json_data_configmap['S3_bucket']['endpoint_url']

                  s3_path = json_data_request['S3_bucket_desc']['folder']
                  #s3_file = json_data_request['S3_bucket_desc'].get('filename',None)

                  def threadentry():
                        app.logger.info('All json data read', extra={'logName': 'json_data_read'})

                        clientS3 = S3Client(aws_access_key_id=s3_access_key, aws_secret_access_key=s3_secret_key,endpoint_url=s3_region_endpoint)
                        clientS3.set_as_default_client()

                        app.logger.info('Client is ready', extra={'logName': 'client_ready'})
                        nonlocal s3_path
                        if s3_path.endswith('/'):
                              s3_path=s3_path[:-1]
                        cp = CloudPath("s3://"+s3_bucket_output+'/'+s3_path+'/', client=clientS3)
                        cpOutput = CloudPath("s3://"+s3_bucket_output+'/result-uc5-cfactor/')
                        app.logger.info("path is s3://"+s3_bucket_output+'/result-uc5-cfactor/', extra={'logName': 'path_output'})

                        with cpOutput.joinpath('log.txt').open('w') as fileOutput:
                              def read_data(folder):
                                    log_function('Opening folder '+str(folder))
                                    with folder.open('rb') as fileBand, rasterio.io.MemoryFile(fileBand) as memfile:
                                          with memfile.open(driver="GTiff",sharing=False) as band_file:
                                                meta=band_file.meta
                                                result=band_file.read().astype(np.float32)/10000.0
                                                app.logger.info('Result obtained', extra={'logName': 'result_obtained'})
                                                return result,meta
                              
                              to_treat={}
                              for folder in cp.iterdir():
                                    if folder.name.endswith('.tiff') or folder.name.endswith('.tif'):
                                          data=read_data(folder)
                                          app.logger.info('datashape '+str(data[0].shape), extra={'logName': 'datashape'})
                                          to_treat[folder.name]=data
                              for key,(value,meta) in to_treat.items():
                                    shapeArray=value.shape
                                    xshape=shapeArray[1]
                                    yshape=shapeArray[2]
                                    app.logger.info('shape '+str(xshape)+' '+str(yshape), extra={'logName': 'shape'})
                                    resultArray=np.zeros([xshape,yshape],dtype=np.float32)
                                    count=np.zeros([xshape,yshape],dtype=np.float32)

                                    toInfer=[]
                                    for i in range(0,xshape-8,9):
                                          for j in range(0,yshape-8,9):
                                                subarray=value[0:3,i:i+9,j:j+9]
                                                dic={}
                                                dic["data"]=np.expand_dims(subarray.astype(np.float32),axis=0)
                                                dic["i"]=i
                                                dic["j"]=j
                                                toInfer.append(dic)
                                          if yshape%9!=0:
                                                j=yshape-9
                                                subarray=value[:,i:i+9,j:yshape]
                                                dic={}
                                                dic["data"]=np.expand_dims(subarray.astype(np.float32),axis=0)
                                                dic["i"]=i
                                                dic["j"]=j
                                                toInfer.append(dic)
                                    if xshape%9!=0:
                                          i=xshape-9
                                          for j in range(0,yshape-8,9):
                                                subarray=value[0:3,i:xshape,j:j+9]
                                                dic={}
                                                dic["data"]=np.expand_dims(subarray.astype(np.float32),axis=0)
                                                dic["i"]=i
                                                dic["j"]=j
                                                toInfer.append(dic)
                                          if yshape%9!=0:
                                                j=yshape-9
                                                subarray=value[0:3,i:xshape,j:j+9]
                                                dic={}
                                                dic["data"]=np.expand_dims(subarray.astype(np.float32),axis=0)
                                                dic["i"]=i
                                                dic["j"]=j
                                                toInfer.append(dic)
                                    app.logger.info('start inference', extra={'logName': 'start_inference'})
                                    app.logger.info('length '+str(len(toInfer)), extra={'logName': 'length'})
                                    asyncio.run(doInference(toInfer,log_function))
                                    app.logger.info('inference done', extra={'logName': 'inference_done'})
                                    for requestElem in toInfer:
                                          result_subarray=requestElem["result"]
                                          i=requestElem["i"]
                                          j=requestElem["j"]
                                          
                                          resultArray[i+0:i+9,j+0:j+9]=resultArray[i+0:i+9,j+0:j+9]+result_subarray
                                          count[i+0:i+9,j+0:j+9]=count[i+0:i+9,j+0:j+9]+1.0

                                    app.logger.info('array all done', extra={'logName': 'array_all_done'})
                                    resultArray=resultArray/count

                                    transform=rasterio.transform.AffineTransformer(meta['transform'])

                                    rows, cols = resultArray.shape

                                    xArray, yArray = np.indices((rows, cols))
                                    xflat = xArray.flatten()
                                    yflat = yArray.flatten()
                                    resultflat = resultArray.flatten()

                                    combined = np.vstack((xflat, yflat, resultflat)).T

                                    df = pd.DataFrame(combined, columns=['x [m]', 'y [m]', 'cfactor'])
                                    outputPath=cpOutput.joinpath(key+'-cfactor-result.csv')
                                    app.logger.info('csv writting', extra={'logName': 'csv_writting'})
                                    with outputPath.open('w') as outputFile:
                                          df.to_csv(outputFile, index=False,header=True)
                                    app.logger.info('csv writting done', extra={'logName': 'csv_writting_done'})
                                    jsonData={}
                                    jsonData['data']=resultArray.tolist()
                                    jsonData['shape']=resultArray.shape
                                    jsonData['type']=str(resultArray.dtype)
                                    meta['crs']=meta['crs'].to_string()
                                    jsonData['metadata']=meta

                                    outputPath=cpOutput.joinpath(key+'-cfactor-result.json')
                                    app.logger.info('start json writting', extra={'logName': 'start_json_writting'})
                                    with outputPath.open('w') as outputFile:
                                          json.dump(jsonData, outputFile)
                                    app.logger.info('json writting done', extra={'logName': 'json_writting_done'})
                                    outputPath=cpOutput.joinpath(key+'-cfactor-result.tiff')
                                    app.logger.info('start tiff writting', extra={'logName': 'start_tiff_writting'})
                                    with outputPath.open('wb') as outputFile, rasterio.io.MemoryFile() as memfile:
                                          app.logger.info('height '+str(xshape)+' width '+str(yshape), extra={'logName': 'height_width'})
                                          app.logger.info('type height '+str(type(xshape))+' type width '+str(type(yshape)), extra={'logName': 'type_height_width'})
                                          app.logger.info('crs '+str(meta['crs']), extra={'logName': 'crs'})
                                          with memfile.open(driver="GTiff",crs=meta['crs'],transform=meta['transform'],height=xshape,width=yshape,count=1,dtype=resultArray.dtype) as dst:
                                                dst.write(resultArray,1)
                                                dst.set_band_description(1,'cfactor, value between 0 and 1 indicating the risk of erosion')
                                                dst.set_band_unit(1,'Unitless')
                                          outputFile.write(memfile.read())
                                    app.logger.info('tiff writting done', extra={'logName': 'tiff_writting_done'})
                              
                              app.logger.info('Output written', extra={'logName': 'output_written'})
                              app.logger.info('Connecting to Kafka', extra={'logName': 'connecting_kafka'})
      
                              response_json ={
                              "previous_component_end": "True",
                              "S3_bucket_desc": {
                                    "folder": "result-uc6-classifier","filename": ""
                              },
                              "meta_information": json_data_request.get('meta_information',{})}
                              Producer.send(kafka_out,key='key',value=response_json)
                              Producer.flush()
                  thread = threading.Thread(target=threadentry)
                  thread.start()
                  response = make_response({
                              "msg": "Started the process"
                              })

            except Exception as e:
                  app.logger.error('Got exception '+str(e), extra={'logName': 'exception'})
                  app.logger.error(traceback.format_exc(), extra={'logName': 'exception'})
                  app.logger.info('So we are ignoring the message', extra={'logName': 'ignore'})
                  # HTTP answer that the message is malformed. This message will then be discarded only the fact that a sucess return code is returned is important.
                  response = make_response({
                  "msg": "There was a problem ignoring"
                  })
            return response

      # This function is used to do the inference on the data.
      # It will connect to the triton server and send the data to it.
      # The result will be returned.
      # The data should be a numpy array of shape (1,10,120,120) and type float32.
      # The result will be a json with the following fields:
      # model_name : The name of the model used.
      # outputs : The result of the inference.
      async def doInference(toInfer,log_function):

            triton_client = httpclient.InferenceServerClient(url="default-inference.uc5.svc.ecmwf-inference-server.local", verbose=False,conn_timeout=10000000,conn_limit=None,ssl=False)
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
                              inputs.append(httpclient.InferInput('input',toInfer[count]["data"].shape, "FP32"))
                              inputs[0].set_data_from_numpy(toInfer[count]["data"], binary_data=True)
                              outputs.append(httpclient.InferRequestedOutput('output', binary_data=True))
                              results = await triton_client.infer('cfactor',inputs,outputs=outputs)
                              return (task,results)
                        if task[0]==255:
                              count=task[1]
                              inputs = []
                              outputs = []
                              input=np.zeros([255,toInfer[count]["data"].shape[1],toInfer[count]["data"].shape[2],toInfer[count]["data"].shape[3]],dtype=np.float32)
                              for i in range(0,255):
                                    input[i]=toInfer[count+i]["data"][0]
                              inputs.append(httpclient.InferInput('input',input.shape, "FP32"))
                              inputs[0].set_data_from_numpy(input, binary_data=True)
                              outputs.append(httpclient.InferRequestedOutput('output', binary_data=True))
                              results = await triton_client.infer('cfactor',inputs,outputs=outputs)
                              return (task,results)
                  except Exception as e:
                        app.logger.error('Got exception '+str(e), extra={'logName': 'exception'})
                        app.logger.error(traceback.format_exc(), extra={'logName': 'exception'})
                        nonlocal last_throw
                        last_throw=time.time()
                        return await consume(task)
            
            async def postprocess(task,results):
                  if task[0]==1:
                        result=results.as_numpy('output')[0][0]
                        toInfer[task[1]]["result"]=result
                  if task[0]==255:
                        result=results.as_numpy('output')
                        for i in range(0,255):
                              toInfer[task[1]+i]["result"]=result[i][0]
                        

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
                        app.logger.info('done instance '+str(nb_done_instance)+'Inference done value '+str(nb_InferenceDone)+' postprocess done '+str(nb_Postprocess)+ ' created '+str(nb_Created), extra={'logName': 'progress'})
            while nb_InferenceDone-nb_Created>0 or nb_Postprocess-nb_InferenceDone>0:
                  await asyncio.sleep(0)
            await asyncio.gather(*list_task,*list_postprocess)
            app.logger.info('Inference done','inference_done')
            await triton_client.close()
      return app
      