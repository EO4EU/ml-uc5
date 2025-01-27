import logging
import os

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
    def __init__(self, level=logging.NOTSET,defaultproducer=None):
        super().__init__(level)
        self.producer=defaultproducer

    def emit(self, record):
        message={}
        message["component_name"]=record.source
        message["workflow_name"]=record.workflow_name
        message["status"]=record.status
        message["description"]=record.msg
        message["timestamp"]=record.created
        optional={}
        optional["namespace"]=get_current_namespace()
        optional["pod"]=get_current_pod_name()
        message["optional"]=optional
        if hasattr(record, 'producer'):
            producer=record.producer
        else:
            producer=self.producer
        producer.send("monitoring.notify",key='key',value=message)
        producer.flush()
