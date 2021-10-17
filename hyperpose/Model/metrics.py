import time
import numpy as np

class AvgMetric:
    def __init__(self,name="default_name",init_value=0):
        self.name=name
        self.step=0
        self.value=init_value
    
    def update(self,value):
        self.step+=1
        self.value+=value
        return self.value
    
    def reset(self):
        self.step=0
        self.value=0

    def gen_report_value(self):
        if(self.step==0):
            return 0
        else:
            return self.value/self.step

    def report_train(self):
        report_value=self.gen_report_value()
        msg=f"{self.name}: {report_value:.8f}"
        self.reset()
        return msg

class TimeMetric:
    def __init__(self):
        self.init_time=time.time()
        self.start_time=time.time()
    
    def start_timing(self):
        self.start_time=time.time()
    
    def report_timing(self):
        last_time=self.start_time
        cur_time=time.time()
        self.start_time=cur_time
        return cur_time-last_time

class MetricManager:
    def __init__(self,debug=False):
        self.debug=debug
        self.metric_group={}
        self.metric_name_list=[]
        self.timer=TimeMetric()
    
    def debug_print(self,msg):
        if(self.debug):
            print(msg)

    def update(self,metric_name,metric_value):
        if(type(metric_value)!=np.ndarray and type(metric_value)!=float):
            metric_value = metric_value.numpy()
        if(metric_name not in self.metric_group):
            self.metric_group[metric_name]=AvgMetric(name=metric_name,init_value=0)
            self.metric_name_list.append(metric_name)
        self.debug_print(f"test metric_name:{metric_name} type(metric_value):{type(metric_value)} value:{metric_value}")
        self.metric_group[metric_name].update(value=metric_value)
    
    def report_train(self):
        msg=""
        for midx,metric_name in enumerate(self.metric_name_list,start=1):
            metric=self.metric_group[metric_name]
            msg+=metric.report_train()+" "
            if(midx%3==0 and midx!=0):
                msg+="\n"
        msg.replace("\n\n","\n")
        return msg
    
    def start_timing(self):
        self.timer.start_timing()
    
    def report_timing(self):
        msg=""
        msg+=f"time:{self.timer.report_timing():.8f}"
        return msg
