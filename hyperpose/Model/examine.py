import numpy as np
from .base_model import BasicModel

def exam_model_weights(model:BasicModel):
    weight_list = model.all_weights
    # construct weight_dict
    weight_dict = {}
    for weight in weight_list:
        name = weight.name
        weight_dict[name]=weight
    # exam by name_list
    name_list = sorted(list(weight_dict.keys()))
    for name in name_list:
        shape = weight_dict[name].shape
        print(f"model weight name: {name}\t shape:{shape}")

def exam_npz_dict_weights(npz_dict_path):
    npz_dict = np.load(npz_dict_path, allow_pickle=True)
    # consturct weight_dict
    weight_dict = npz_dict
    # exam by name_list
    name_list = sorted(list(weight_dict.keys()))
    for name in name_list:
        shape = weight_dict[name].shape
        print(f"npz_dict weight name: {name}\t shape:{shape}")

def exam_npz_weights(npz_path):
    print("weights in npz file don't have names")
    npz = np.load(npz_path, allow_pickle=True)
    npz = npz["params"]
    for widx,weight in enumerate(npz):
        print(f"npz weight idx: {widx}\t shape:{weight.shape}")