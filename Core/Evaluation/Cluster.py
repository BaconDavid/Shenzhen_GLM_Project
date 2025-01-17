# %% import packages
import sys
sys.path.append('/mnt/yizhou/Shenzhen_GLM_Project/Core/')

from Utils.utils import get_data_logger, load_esm_embedding
from typing import Literal, get_args

import pandas as pd
# %% preparation for analysis
logger = get_data_logger('Cluster')

embedding_path = '/mnt/yizhou/Data/Preparation_Data/total_esm2_embedding.pkl'
esm2_total_embedding = load_esm_embedding(embedding_path) # load embeddings from ESM model
label_originate = Literal['Interpro'] #where labels come from

# %% get label
class ClusterLabel:
    """
    Possible labels from different sources
    The label format might be tsv or csv, etc.. They must be pre-processed in jupyter notebook and only kept useful columns.
    The sequences used for clustering should be filtered in advance. Eg: Only use pfam database, so tsv only includes sequences annoated by pfam.
    """
    def __init__(self, label_method: label_originate,label_file_path: str):
        if label_method not in get_args(label_originate):
            raise ValueError(f"Label method {label_method} is not supported")
        self.label_file_path = label_file_path

    def get_labels(self,label_lst: tuple):
        """
        Get labels from certain columns
        """
        label_df = self._get_label_df()
        return label_df[list(label_lst)] # get picked columns
        
    def _get_label_df(self,type='tsv'):
        if type == 'tsv':
            label_df = pd.read_csv(self.label_file_path, sep='\t')
        elif type == 'csv':
            label_df = pd.read_csv(self.label_file_path)
        
        return label_df
    
    @property
    def label_df(self):
        return self._get_label_df()

# %%
class ClusterEmbedding:
    def __init__(self,labels,embeddings):
        pass

    @classmethod
    def extract_from_picked_sequences(cls,labels,embeddings,picked_sequences):
        pass


# %% test tsv
cluster_label = ClusterLabel('Interpro','/mnt/yizhou/Data/Label/total_fasta_interpro.tsv')
esm_embedding_labels, esm_embedding = load_esm_embedding(embedding_path)

cluster_labels_df = cluster_label.get_labels(('Signature description'))
picked_sequences = cluster_labels_df.loc[:,'Protein accession']




# %%
