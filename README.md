# Running ESM2/3 models

## ESM2
Directly load the model's weight

## ESM3
Since this model is not open yet (need to apply for authority), in the source code you have to download it remotely (You may also upload from local but I found there was some problem with the path). 

Due to the limitation of network :face_with_head_bandage:, the model weight is downloaded and stored locally. Besides, you have to clone the ESM3 locally and change the path in source code where you store the weight. The code is in **/mnt/yizhou/git_projects/esm/esm/utils/constants/data_root function** You have to change the path from default value of **esm/esm3/** to your weight path.

After change the path, you can directly load the model with the code:

``` python
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein
from esm.sdk.api import ESMProtein

model = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")

# inference to get embedding
protein = ESMProtein(
    sequence=(
        "FIFLALLGAAVAFPVDDDDKIVGGYTCGANTVPYQVSLNSGYHFCGGSLINSQWVVSAAHCYKSGIQVRLGEDNINVVEG"
        "NEQFISASKSIVHPSYNSNTLNNDIMLIKLKSAASLNSRVASISLPTSCASAGTQCLISGWGNTKSSGTSYPDVLKCLKAP"
        "ILSDSSCKSAYPGQITSNMFCAGYLEGGKDSCQGDSGGPVVCSGKLQGIVSWGSGCAQKNKPGVYTKVCNYVSWIKQTIASN"
    )
)
protein_tensor = test.encode(protein)
#protein_tensor.sequence
model = model.to(torch.float32)
#output = test(sequence_tokens = protein_tensor.sequence)

embeddings = model(sequence_tokens = protein_tensor.sequence.unsqueeze(0)).embeddings

```
## Getting embeddings

Run the embedding infernece file: 
For example: Run ESM2:
``` python
python esm_embedding_inference.py --ESM_model_type ESM3 --ESM_model_name esm3_sm_open_v1  --fasta_file /mnt/yizhou/Data/Preparation_Data/Sampled_one_fasta.fasta --output_pkl /mnt/yizhou/Data/Preparation_Data/test_esm.pkl --device cuda
```
**Attention:** When running ESM3 you need to activate a new virtual environment for ESM3 inference, as process of ESM3 inference is different from ESM2, but they have the same package name. 