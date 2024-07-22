#%% importing packages
import torch
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap
import esm
from esm import FastaBatchedDataset
from tqdm import tqdm

import sys
sys.path.append('/mnt/yizhou/Shenzhen_GLM_Project/Core/')

from Utils.utils import get_data_logger
import pickle as pk
from typing import Literal,get_args

# %% get logger
logger = get_data_logger('ESM2_Inference')
logger.info(len(sys.argv))

# %% get inputs pars
fasta_file = sys.argv[1]
output_pkl = sys.argv[2]
ESM_model = sys.argv[3]

# %% ESM model type
ESM_model_name = Literal['ESM2','ESM3']
# %% get model
class EsmModel:
    ESM_model_parameters = {
        'ESM2': ('esm2_t33_650M_UR50D.pt', 'esm2_t33_650M_UR50D-contact-regression.pt')
    }  # model name: (model, regression)

    def __init__(self, ESM_model_type: ESM_model_name = 'ESM2'):
        if ESM_model_type not in get_args(ESM_model_name):
            raise ValueError(f"ESM model {ESM_model_type} is not supported")
        self.ESM_model_type = ESM_model_type

    def get_model_data(self):
        """
        Pre-trained model of ESM2. Manually download the model from the hub and load it.
        """
        model_data = torch.load(
            f"{torch.hub.get_dir()}/checkpoints/{self.ESM_model_parameters[self.ESM_model_type][0]}",
            map_location="cpu",
        )
        regression_data = torch.load(
            f"{torch.hub.get_dir()}/checkpoints/{self.ESM_model_parameters[self.ESM_model_type][1]}",
            map_location="cpu",
        )
        return model_data, regression_data


# %% Get model
esm_model = EsmModel('ESM2')
model_name = 'esm2_t33_650M_UR50D'  # for load model
model_data, regression_data = esm_model.get_model_data()

# %% set distributed backend
url = "tcp://localhost:23456"
torch.distributed.init_process_group(backend="nccl", init_method=url, world_size=1, rank=0)
fsdp_params = dict(
    mixed_precision=True,
    flatten_parameters=True,
    state_dict_device=torch.device("cpu"),  # reduce GPU
)
# %% data preparation
toks_per_batch = 12290
dataset = FastaBatchedDataset.from_file(fasta_file)
batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)

# %% model preparation
with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
    model, vocab = esm.pretrained.load_model_and_alphabet_core(
        model_name, model_data, regression_data
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=vocab.get_batch_converter(), batch_sampler=batches
    )
    model.eval()

    # Wrap each layer in FSDP separately
    for name, child in model.named_children():
        if name == "layers":
            for layer_name, layer in child.named_children():
                wrapped_layer = wrap(layer)
                setattr(child, layer_name, wrapped_layer)
    model = wrap(model)

# %% where to store results
sequence_representations = []

# %% infer embedding from ESM2
with torch.no_grad():
    for batch_idx, (labels, strs, toks) in tqdm(enumerate(data_loader), total = len(data_loader)):
        toks = toks.cuda()
        logger.info(f"shape of tks{toks.shape}")
        toks = toks[:, :12288] #truncate 
        results = model(toks, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]
        for i, label in enumerate(labels):
            truncate_len = min(12288, len(strs[i]))
            sequence_representations.append((label,token_representations[i, 1 : truncate_len + 1].mean(0).detach().cpu().numpy()))

f = open(output_pkl, "wb")
pk.dump(sequence_representations,f)
f.close()
# %%
with open(output_pkl, "rb") as f:
    sequence_representations = pk.load(f)

# %%
if __name__ == '__main__':
    esm_model = EsmModel('ESM2')
    