# %% only for esm3
import sys
sys.path.append('/mnt/yizhou/git_projects/esm')
# %% 
from esm.models import esm3
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
from esm.utils.constants.esm3 import data_root
import torch
from tqdm import tqdm
from Utils.Data.esm3_batch_inference import FastaBatchedDataset,BatchConverter

# %% constant parameters
tok_per_batch = 12290
class ESM:
    def __init__(self,model_name:str = 'esm3_sm_open_v1'):
        self.model_name = model_name

    def get_model(self):
        return ESM3.from_pretrained("esm3_sm_open_v1")

    def get_dataloader(self,fasta_file):
        dataset = FastaBatchedDataset.from_file(fasta_file)
        batches = dataset.get_batch_indices(tok_per_batch, extra_toks_per_seq=1)
        data_loader = torch.utils.data.DataLoader(
                dataset, collate_fn= BatchConverter(), batch_sampler=batches
            )
        return data_loader



