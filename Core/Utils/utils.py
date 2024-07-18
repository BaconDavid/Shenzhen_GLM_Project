import os
import logging
from functools import wraps

#check if path is created by index or par name.
def path_check(path_param_index=0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 获取路径参数
            path = args[path_param_index] if path_param_index < len(args) else kwargs.get(path_param_index)
            
            # 检查并创建路径
            if path and not os.path.exists(path):
                os.makedirs(path)
                print(f'Created the {path} directory')
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def get_data_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    fh = logging.FileHandler(f'{name}.log',mode='a')
    fh.setLevel(logging.DEBUG)  # 修正此行
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger