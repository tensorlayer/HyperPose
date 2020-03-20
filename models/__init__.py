import tensorflow as tf
from .inference.common import rename_tensor


__all__ = [
    'get_model'
]

def get_model(model_type,config):
    MODEL=config.MODEL
    if model_type == "lightweight_openpose":
        print(f"using model light weight openpose!")
        from .lightweight_openpose import model
        ret_model=model(n_pos=MODEL.n_pos,num_channels=MODEL.num_channels,hin=MODEL.hin,win=MODEL.win,\
            hout=MODEL.hout,wout=MODEL.wout)
    elif model_type == "pose_proposal":
        print(f"using model pose proposal network!")
        from .pose_proposal import model
        ret_model=model(K_size=MODEL.K_size,L_size=MODEL.L_size,hnei=MODEL.hnei,wnei=MODEL.wnei,lmd_rsp=MODEL.lmd_rsp,\
            lmd_iou=MODEL.lmd_iou,lmd_coor=MODEL.lmd_coor,lmd_size=MODEL.lmd_size,lmd_limb=MODEL.lmd_limb)
    else:
        raise RuntimeError(f'unknown model type {model_type}')
    return ret_model

def get_train(model_type):
    if model_type == "lightweight_openpose":
        print(f"training light weight openpose...")
        from .lightweight_openpose import single_train,parallel_train
    elif model_type == "pose_proposal":
        print(f"training pose proposal network...")
        from .pose_proposal import single_train,parallel_train
    else:
        raise RuntimeError(f'unknown model type {model_type}')
    return single_train,parallel_train




