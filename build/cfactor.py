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

class ThroughputMeter:
    def __init__(self, report_every=1.0,logger=None,file_number=0,total_number=1,timings=[],total_pixels=1):
        self.report_every = report_every
        self._t0 = time.perf_counter()
        self._last_t = self._t0
        self.total_reqs = 0          # completed Triton calls
        self.total_items = 0         # rows predicted (sum batch sizes)
        self.logger = logger
        self.file_number = file_number
        self.total_number = total_number
        self.timings = timings
        self.total_pixels = total_pixels

        self._last_reqs = 0
        self._last_items = 0
        self._stop = asyncio.Event()
        self._inflight = 0
        self._concurrent = 0

    def update(self, batch_size: int, concurrent: int = 1, inflight: int = 1):
        # call once per COMPLETED Triton inference
        self.total_reqs += 1
        self.total_items += batch_size
        self._concurrent = concurrent
        self._inflight = inflight

    async def reporter(self):
        while not self._stop.is_set():
            await asyncio.sleep(self.report_every)
            now = time.perf_counter()
            dt = now - self._last_t
            if dt <= 0:
                continue
            # windowed rates (since last report)
            reqs_window = self.total_reqs - self._last_reqs
            items_window = self.total_items - self._last_items
            reqs_per_s = reqs_window / dt
            items_per_s = items_window / dt
            # overall averages
            total_dt = now - self._t0
            avg_reqs = self.total_reqs / total_dt if total_dt > 0 else 0.0
            avg_items = self.total_items / total_dt if total_dt > 0 else 0.0
            
            # Current file throughput and time estimation
            current_file_time = now - self._t0
            current_file_throughput = self.total_items / current_file_time if current_file_time > 0 else 0.0
            
            # Estimate time remaining for current file
            current_file_est_msg = ""
            if self.total_pixels > 0 and current_file_throughput > 0:
                pixels_remaining = self.total_pixels - self.total_items
                est_time_current_file = pixels_remaining / current_file_throughput
                progress_pct = (self.total_items / self.total_pixels) * 100
                current_file_est_msg = f" ({progress_pct:.1f}% complete, ~{est_time_current_file:.1f}s remaining)"
            
            # Calculate total time estimation
            time_est_msg = ""
            if len(self.timings) > 0:
                # Have historical data: use average time per file for remaining files + current file pixel estimate
                avg_time_per_file = sum(self.timings) / len(self.timings)
                remaining_files_after_current = self.total_number - self.file_number
                est_remaining_files_time = avg_time_per_file * remaining_files_after_current
                
                # Add current file estimate based on pixels
                if self.total_pixels > 0 and current_file_throughput > 0:
                    pixels_remaining = self.total_pixels - self.total_items
                    est_time_current_file = pixels_remaining / current_file_throughput
                    total_estimated_time = est_time_current_file + est_remaining_files_time
                    time_est_msg = f" | File {self.file_number}/{self.total_number} | Avg time/file: {avg_time_per_file:.1f}s | Est. total remaining: {total_estimated_time:.1f}s"
                else:
                    time_est_msg = f" | File {self.file_number}/{self.total_number} | Avg time/file: {avg_time_per_file:.1f}s | Est. remaining: {est_remaining_files_time:.1f}s"
            else:
                # First file: estimate assuming all files have same number of pixels
                if self.total_pixels > 0 and current_file_throughput > 0:
                    pixels_remaining = self.total_pixels - self.total_items
                    est_time_current_file = pixels_remaining / current_file_throughput
                    est_time_all_files = (est_time_current_file / (self.total_items / self.total_pixels)) * self.total_number if self.total_items > 0 else 0
                    time_est_msg = f" | File {self.file_number}/{self.total_number} | Est. total remaining: {est_time_all_files:.1f}s (pixel-based)"
                else:
                    time_est_msg = f" | File {self.file_number}/{self.total_number}"

            if self.logger:
                  self.logger.info(f"[throughput] {reqs_per_s:.2f} req/s, {items_per_s:.0f} pixels/s  "
                                     f"(avg: {avg_reqs:.2f} req/s, {avg_items:.0f} pixels/s) "
                                     f"| Current file: {current_file_throughput:.0f} pixels/s, {current_file_time:.1f}s elapsed{current_file_est_msg}"
                                     f"{time_est_msg}", 
                                     extra={'status': 'INFO','overwrite':True})

            self._last_t = now
            self._last_reqs = self.total_reqs
            self._last_items = self.total_items

    def stop(self):
        self._stop.set()

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
                                    listFiles=[]
                                    def list_files(cp):
                                          for path in cp.iterdir():
                                                pattern=r'.*MSIL2A.*\.SAFE$'
                                                match = re.search(pattern,path.name)
                                                if match:
                                                      listFiles.append(path)
                                                list_files(path)
                                    list_files(cp)
                                    file_timings=[]
                                    total_number=len(listFiles)
                                    for file_index, folder in enumerate(listFiles, start=1):
                                          file_start_time=time.time()
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
                                                                  if matchedBand in ['B05','B06','B07','B8A','B11','B12','SCL']:
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
                                                      "SCL"
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
                                                      bands_data['B3']=bands_data['B03']
                                                      bands_data['B4']=bands_data['B04']
                                                      bands_data['B2']=bands_data['B02']
                                                      bands_data['B8']=bands_data['B08']
                                                      del bands_data['B03']
                                                      del bands_data['B04']
                                                      del bands_data['B02']
                                                      del bands_data['B08']
                                                      BANDS_ALL=['B11','B12','B2','B3','B4','B8','SCL']
                                                      def data_generator():
                                                            listvalue=[]
                                                            for i in range(0,h):
                                                                  for j in range(0,w):
                                                                        dic={}
                                                                        for band_name in BANDS_ALL:
                                                                              band_data=bands_data[band_name]
                                                                              dic[band_name]=band_data[i,j]
                                                                        dic['x']=i
                                                                        dic['y']=j
                                                                        listvalue.append(dic)
                                                                        if len(listvalue)>=10000:
                                                                              yield listvalue
                                                                              listvalue=[]
                                                            if len(listvalue)>0:
                                                                  yield listvalue
                                                                  listvalue=[]

                                                      def process(listvalue):
                                                            value=pd.DataFrame(listvalue)
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

                                                            # === Apply robust scaling ===
                                                            # During training, all features were scaled using robust scaling (no time features):
                                                            # scaled = (x - q1) / IQR
                                                            X_scaled = X.copy()
                                                            for i in range(X.shape[1]):  # Scale all features
                                                                  fname = list(scaler_params['iqr'].keys())[i]
                                                                  q1 = scaler_params['q1'][fname]
                                                                  iqr = scaler_params['iqr'][fname]
                                                                  X_scaled[:, i] = (X[:, i] - q1) / iqr
                                                            return X_scaled, value[['x']].values, value[['y']].values, value[['SCL']].values
                                                      async def do_inference(data,sem,triton_client):
                                                            # Prepare inputs and outputs in a separate thread to avoid blocking
                                                            def prepare_inference_inputs(data):
                                                                  inputs = []
                                                                  outputs = []
                                                                  data = data.astype(np.float32)
                                                                  
                                                                  inputs.append(httpclient.InferInput('input__0', data.shape, "FP32"))
                                                                  inputs[0].set_data_from_numpy(data, binary_data=True)
                                                                  outputs.append(httpclient.InferRequestedOutput('output__0', binary_data=True))
                                                                  
                                                                  return inputs, outputs
                                                            
                                                            # Run input preparation in thread
                                                            inputs, outputs = await asyncio.to_thread(prepare_inference_inputs, data)
                                                            
                                                            # Only the actual inference needs the semaphore
                                                            async with sem:
                                                                  results = await triton_client.infer('cfactor2', inputs, outputs=outputs)
                                                            
                                                            # Make numpy result conversion async to avoid blocking event loop
                                                            return await asyncio.to_thread(lambda: results.as_numpy('output__0'))

                                                      async def handle_one(data,sem,triton_client):
                                                            # Make the heavy numpy processing async to avoid blocking event loop
                                                            v1,v2,v3,v4 = await asyncio.to_thread(process, data)
                                                            try:
                                                                  result = await do_inference(v1,sem,triton_client=triton_client)+target_mean - 0.5
                                                                  
                                                                  # Move post-processing numpy operations to thread to avoid blocking
                                                                  def post_process_result(result, v4):
                                                                        def is_cloud(scl):
                                                                              # SCL values indicating cloud or cloud shadow
                                                                              return scl in [3, 8, 9, 10]
                                                                        def is_water(scl):
                                                                              # SCL value indicating water
                                                                              return scl == 6
                                                                        def no_data_or_invalid(scl):
                                                                              # SCL values indicating no data or invalid data
                                                                              return scl in [0, 1, 7]
                                                                        def is_ice(scl):
                                                                              # SCL value indicating snow or ice
                                                                              return scl == 11
                                                                        v4 = v4.flatten()
                                                                        result = np.where(np.vectorize(is_cloud)(v4), -1, result)
                                                                        result = np.where(np.vectorize(is_water)(v4), 0, result)
                                                                        result = np.where(np.vectorize(no_data_or_invalid)(v4), -1, result)
                                                                        result = np.where(np.vectorize(is_ice)(v4), 0, result)
                                                                        return result
                                                                  
                                                                  result = await asyncio.to_thread(post_process_result, result, v4)
                                                            except Exception as e:
                                                                  await asyncio.sleep(1)
                                                                  return await handle_one(data,sem,triton_client=triton_client)
                                                            return (result,v2,v3)

                                                      async def run_pipeline(max_concurrent_tasks=10,max_in_flight=60,file_number=0,total_number=1,timings_file=[],total_pixels=1):
                                                            sem = asyncio.Semaphore(max_concurrent_tasks)
                                                            tasks = set()
                                                            meter = ThroughputMeter(report_every=60.0, logger=logger_workflow, file_number=file_number, total_number=total_number, timings=timings_file, total_pixels=total_pixels)
                                                            reporter_task = asyncio.create_task(meter.reporter())
                                                            array=np.zeros((h,w),dtype=np.float32)
                                                            triton_client = httpclient.InferenceServerClient(url="default-inference.uc5.svc.cineca-inference-server.local", verbose=False,conn_timeout=10000000,conn_limit=None,ssl=False)
                                                            try:
                                                                  for data in data_generator():
                                                                        t = asyncio.create_task(handle_one(data,sem,triton_client=triton_client))
                                                                        tasks.add(t)

                                                                        if len(tasks) >= max_in_flight:
                                                                              _done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                                                                              for fut in _done:
                                                                                    result,x,y = fut.result()
                                                                                    meter.update(batch_size=result.shape[0], concurrent=max_concurrent_tasks-sem._value, inflight=len(tasks))
                                                                                    for i in range(0,result.shape[0]):
                                                                                          result_subarray=result[i]
                                                                                          array[x[i,0],y[i,0]]=result_subarray
                                                                  if tasks:
                                                                        _done, tasks = await asyncio.wait(tasks)
                                                                        for fut in _done:
                                                                              result,x,y = fut.result()
                                                                              meter.update(batch_size=result.shape[0], concurrent=max_concurrent_tasks-sem._value, inflight=len(tasks))
                                                                              for i in range(0,result.shape[0]):
                                                                                    result_subarray=result[i]
                                                                                    array[x[i,0],y[i,0]]=result_subarray
                                                            finally:
                                                                  meter.stop()
                                                                  await reporter_task
                                                                  await triton_client.close()
                                                            logger_workflow.debug(f"[summary] total calls: {meter.total_reqs}, total items: {meter.total_items}", extra={'status': 'DEBUG'})
                                                            return array
                                                      logger_workflow.debug('start processing', extra={'status': 'DEBUG'})
                                                      array = asyncio.run(run_pipeline(file_number=file_index, total_number=total_number, timings_file=file_timings,total_pixels=h*w))
                                                      logger_workflow.debug('processing done', extra={'status': 'DEBUG'})

                                                      outputPath=cpOutput.joinpath(folder.name+"_cfactor.tiff")
                                                      logger_workflow.debug('start writing output to '+str(outputPath), extra={'status': 'DEBUG'})
                                                      with outputPath.open('wb') as outputFile,rasterio.io.MemoryFile() as memfile:
                                                            #with rasterio.open(outputFile,mode='w',**data["meta"][ALL_BANDS[band_number]]) as file2:
                                                            with memfile.open(driver="GTiff",width=w,height=h,count=1,dtype="float32",crs=metaData["B03"]["crs"],transform=metaData["B03"]["transform"],compress='ZSTD',nodata=-1) as file2:
                                                                  file2.write(array, indexes=1)
                                                            outputFile.write(memfile.read())
                                          file_end_time=time.time()
                                          file_timings.append(file_end_time - file_start_time)
                                    
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

      return app
      