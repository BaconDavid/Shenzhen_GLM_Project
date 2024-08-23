# %%
import sys
import argparse
from typing import Sequence, List

from sympy import sequence
import torch
from tqdm import tqdm
from Utils.utils import path_check
import pickle as pk

ESM_MODEL_TYPE = ['ESM2','ESM3']
ESM_MODEL_DICT = {
    "ESM2": ['esm2_t33_650M_UR50D'],
    "ESM3": ['esm3_sm_open_v1'],

}

class EsmModel:
    def __init__(self,esm_model_type: str = 'ESM2',esm_model_name:str = None):
        if esm_model_type not in ESM_MODEL_TYPE:
            raise ValueError(f"Invalid ESM model type: {esm_model_type}. Must be one of {ESM_MODEL_TYPE}")
        if esm_model_name not in ESM_MODEL_DICT[esm_model_type]:
            raise ValueError(f"Invalid model name: {esm_model_name} for model type: {esm_model_type}. Must be one of {ESM_MODEL_DICT[esm_model_type]}")
        self.esm_model_type = esm_model_type
        self.esm_model_name = esm_model_name

        if self.esm_model_type == 'ESM2':
            from Embedding.ESM2_embedding import ESM
        elif self.esm_model_type == 'ESM3':
            from Embedding.ESM3_embedding import ESM

        self.esm = ESM(self.esm_model_name)

    def get_esm_model(self):
        return self.esm.get_model()

    def get_esm_dataloader(self,fasta_file):
        return self.esm.get_dataloader(fasta_file)
    
class EsmEmbeddingInference:
    def __init__(self,model,data_loader,device,esm_model_type: str = 'ESM2'):
        self.model = model
        self.data_loader = data_loader
        self.esm_model_type = esm_model_type
        self.device = device
    
    def embedding_inference(self):
        if self.esm_model_type == 'ESM2':
            return self._esm2_embedding_inference()
        elif self.esm_model_type == 'ESM3':
            return self._esm3_embedding_inference()

    def _esm2_embedding_inference(self):
        sequence_representations = []
        self.model.to(self.device)
        #infernce
        with torch.no_grad():
                for batch_idx, (labels, strs, toks) in tqdm(enumerate(data_loader), total = len(data_loader)):
                    toks = toks.to(self.device)
                    toks = toks[:, :12288] #truncate 
                    results = model(toks, repr_layers=[33], return_contacts=False)
                    token_representations = results["representations"][33]
                    for i, label in enumerate(labels):
                        truncate_len = min(12288, len(strs[i]))
                        sequence_representations.append((label,token_representations[i, 1 : truncate_len + 1].mean(0).detach().cpu().numpy()))
        return sequence_representations

    def _esm3_embedding_inference(self):
        sequence_representations = []
        self.model.to(self.device)
        #same data type
        self.model = self.model.to(torch.float32) # same data type
        #inference
        with torch.no_grad():
            for batch_idx, (labels, strs, toks) in tqdm(enumerate(data_loader), total = len(data_loader)):
                toks = toks.to(self.device)
                toks = toks[:, :12288] #truncate 
                results = model(sequence_tokens = toks)
                token_representations = results.embeddings
                print(token_representations,'normalized!')

                for i, label in enumerate(labels):
                    truncate_len = min(12288, len(strs[i]))
                    sequence_representations.append((label,token_representations[i, 1 : truncate_len + 1].mean(0).detach().cpu().numpy()))
        return sequence_representations


def embedding_save(output_pkl:str = None, sequence_representations:List = None):
    print(sequence_representations,666)
    with open(output_pkl, "wb") as f:
        pk.dump(sequence_representations, f)
# %% inference
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ESM model inference script')
    parser.add_argument('--ESM_model_type',default='ESM2', type=str, help='Type of ESM model (e.g., ESM2, ESM3)')
    parser.add_argument('--ESM_model_name', default='esm2_t33_650M_UR50D', type=str, help='Name of the ESM model')
    parser.add_argument('--fasta_file', type=str, help='Path to the input FASTA file')
    parser.add_argument('--output_pkl', type=str, help='Path to the output PKL file')
    parser.add_argument('--device', type=str, help='Device to run the model on (e.g., cpu, cuda)')

    args = parser.parse_args()
    esm_model = EsmModel(args.ESM_model_type,args.ESM_model_name)
    model = esm_model.get_esm_model()
    model.eval()
    model.to(args.device)
    data_loader = esm_model.get_esm_dataloader(fasta_file=args.fasta_file)
    embedding_inference = EsmEmbeddingInference(model,data_loader,args.device,esm_model_type=args.ESM_model_type)
    sequence_representations = embedding_inference.embedding_inference()
    #save embeddings
    embedding_save(args.output_pkl,sequence_representations)






    


# %%
