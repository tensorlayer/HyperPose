import logging

def init_log(config):
    logging.basicConfig(filename=config.LOG.log_path,filemode="a",level=logging.INFO)

def log(msg):
    logging.log(level=logging.INFO,msg=msg)
    print(msg)