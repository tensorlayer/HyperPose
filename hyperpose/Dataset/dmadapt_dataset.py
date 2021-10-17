import tensorflow as tf
from .common import basic_map_func, get_num_parallel_calls
from .common import log_data as log

class Domainadapt_dataset:
    def __init__(self,image_paths):
        self.image_paths=image_paths
        log(f"Domainadapt dataset constructed, total {len(self.image_paths)} adapt images.")
    
    def get_train_dataset(self):
        # tensorflow data pipeline
        def generator():
            """TF Dataset generator."""
            for _input in self.image_paths:
                yield _input.encode('utf-8')
        train_dataset = tf.data.Dataset.from_generator(generator,output_types=tf.string)
        train_dataset = train_dataset.map(map_func=basic_map_func, num_parallel_calls=get_num_parallel_calls())
        return train_dataset
