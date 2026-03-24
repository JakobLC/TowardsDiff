# Towards Diff Repo
The official github repo for the paper "Towards Agnostic and Holistic Universal Image Segmentation with Bit Diffusion" at NLDL26. The paper is available at [[arxiv]](https://arxiv.org/abs/2601.02881).

# Environment Setup

To create an environment suitable for running the code, you can use the following commands (maybe replace torch install with compatible cuda version):
```bash
conda create -n td-env python=3.12
conda activate td-env
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

# Data 

To download the Entity dataset visit this [google drive](https://drive.google.com/drive/folders/1ghgYcEoCsbtSoQlMfD7-zUahYraU01zn) and unzip each of the three files in `data/entity/`. Additionally, download `entityseg_train_lr.json` and `entityseg_val_lr.json` from the official [github repo](https://github.com/adobe-research/EntitySeg-Dataset/releases/tag/v1.0) and place both files in `data/entity/`. Then run:

```bash
python data/entity/process_entity.py
```

This generates `data/entity/val_ims.txt`, `data/entity/test_ims.txt`, and per-folder mask images in `data/entity/entity_*_masks/*_mask.png`.

# Training and Sampling

Afterwards, training is done using:
```
python train.py
```
To specify a model id, use `--model_name name[version]`. Models from the paper:
% table with columns: model_name arg, description

| `model_name` | description | #train runs |
| --- | --- | --- |
| `analog_bits` | Diffusion on analog bits (Our model). | 1 |
| `analog_bits[lap_sweep]` | Our model sweeping over the type of Location Aware Pallete (LAP). | 4 |
| `analog_bits[pred_loss_weights_sweep]` | Our model sweeping over both the loss weights and prediction type. | 12 |
| `analog_bits[pred_sweep]` | Our model sweeping over the prediction types with sigmoid_-4 loss weights. | 3 |
| `analog_bits[pred_sweep]` | Our model sweeping over bias in sigmoid loss weights. | 5 |
| `analog_bits[size_sweep]` | Our model sweeping over number of agnostic classes. | 5 |
| `unigs` | Using RGB encoding from UNIGS. | 1 |
| `unigs[pred_sweep]` | Using RGB encoding from UNIGS and sweeping over prediction types | 3 |
| `onehot` | Using onehot encoding. | 1 |
| `onehot[pred_sweep]` | Using onehot encoding and sweeping over prediction types | 3 |
| `onehot[act_sweep]` | Using onehot encoding and sweeping over activation functions. | 3 |
| `onehot[size_sweep]` | Using onehot encoding and sweeping over number of agnostic classes. | 5 |

After training, sampling can be done using:

```
python sample.py --name_match_str *analog_bits_0
```
which will run sampling for all ckpts matching the pattern `saves/*analog_bits_0/*.pt`. By default, this runs sampling for the `--gen_setup vali` generation setup. The defined config options for all generations setups are shown in the table below:

| `gen_setup` | description | #sampling runs |
| --- | --- | --- |
| `vali` | Validation set evaluation. | 1 |
| `train` | Training set evaluation. | 1 |
| `test` | Test set evaluation. | 1 |
| `sweeps[ts]` | Timestep sweeps (DDPM sampler). | 12 |
| `sweeps[ts][ddim]` | Timestep sweeps with DDIM sampler. | 12 |
| `sweeps[gw]` | Guidance weight sweeps. | 8 |
| `64x3[train]` | 64 image evaluation with 3 predictions per image on the training set (used during training mainly). | 1 |
| `64x3[vali]` | 64 image evaluation with 3 predictions per image on the validation set (used during training mainly). | 1 |

# Citation
If you find this code useful, please consider citing our paper:
```
@article{towards_diff,
      title={Towards Agnostic and Holistic Universal Image Segmentation with Bit Diffusion}, 
      author={Jakob Lønborg Christensen and Morten Rieger Hannemose and Anders Bjorholm Dahl and Vedrana Andersen Dahl},
      year={2026},
      journal={Northern Lights Deep Learning Conference (NLDL26)},
}
```
