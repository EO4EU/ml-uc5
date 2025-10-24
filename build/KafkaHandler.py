import logging
import os
from datetime import datetime, timezone

def get_current_namespace():
    try:
        with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        # Handle the case where the file doesn't exist
        return ''
    
def get_current_pod_name():
    return os.getenv("HOSTNAME", "Unknown")

class DefaultContextFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'status'):
            record.status = 'INFO'
        if not hasattr(record, 'workflow_name'):
            record.workflow_name = ''
        if not hasattr(record, 'source'):
            record.source = 'GenericUC5CFactor'
        return True

class KafkaHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        message={}
        message["component_name"]=record.source
        message["workflow_name"]=record.workflow_name
        message["status"]=record.status
        message["description"]=record.msg[:300]
        timestamp = record.created
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        message["timestamp"]=dt.isoformat()
        optional={}
        optional["namespace"]=get_current_namespace()
        optional["pod"]=get_current_pod_name()
        if hasattr(record,'overwrite') and record.overwrite:
            optional['overwrite']=True
        message["optional"]=optional
        if hasattr(record, 'producer'):
            producer=record.producer
            producer.send("monitoring.notify",key='key',value=message)
            producer.flush()
