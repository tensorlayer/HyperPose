def get_dataset(dataset_name,config):
    if(dataset_name=="coco"):
        from .coco_dataset import get_dataset
    else:
        raise RuntimeError(f'unknown dataset_name {dataset_name}')
    return get_dataset(config)
