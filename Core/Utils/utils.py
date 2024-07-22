import os
import logging
from functools import wraps
import pickle as pk

#check if path is created by index or par name.
def path_check(path_param_index=0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args):
            # 获取路径参数
            path = args[path_param_index] 
            # 检查并创建路径
            if path and not os.path.exists(path):
                os.makedirs(path)
                print(f'Created the {path} directory')
            
            return func(*args)
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

def load_esm_embedding(pickle_file:str):
    """
    Load esm embedding from pickle file
    """
    esm_embedding, esm_embedding_accession = [], []
    with open(pickle_file, 'rb') as f:
        esm_embedding_all = pk.load(f)
    
    for embeddings_info in esm_embedding_all:
        esm_embedding_accession.append(embeddings_info[0])
        esm_embedding.append(embeddings_info[1])
        
    return esm_embedding, esm_embedding_accession


