import numpy as np

class AvgMetric:
    def __init__(self,name="default_name",init_value=0,metric_interval=100):
        self.step=0
        self.name=name
        self.value=init_value
        self.metric_interval=metric_interval
    
    def update(self,update_value):
        self.step+=1
        self.value+=update_value/self.metric_interval
        return self.value
    
    def reset(self):
        self.value=0

    def get_metric(self):
        msg=f"{self.name}:{self.value}"
        self.reset()
        return msg
