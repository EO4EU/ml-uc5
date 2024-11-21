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
        if not hasattr(record, 'logName'):
            record.logName = ''
        return True

class KafkaHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET,producer=None,source=''):
        super().__init__(level)
        self.source=source
        self.producer=producer

    def emit(self, record):
        message={}
        message["source"]=self.source
        message["name"]=record.logName
        message["level"]=record.levelname
        message["description"]=record.msg
        message["description_business"]=record.msg
        message["timestamp"]=record.created
        optional={}
        optional["namespace"]=get_current_namespace()
        optional["pod"]=get_current_pod_name()
        message["optional"]=optional
        self.producer.send("monitoring.notify",key='key',value=message)
        self.producer.flush()