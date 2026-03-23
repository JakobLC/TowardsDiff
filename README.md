The official github repo for the paper "Towards Agnostic and Holistic Universal Image Segmentation with Bit Diffusion" at NLDL26. The paper is available at [[arxiv]](https://arxiv.org/abs/2601.02881).

To create an environment suitable for running the code, you can use the following commands (maybe replace with compatible cuda version):
```bash
conda create -n td-env python=3.12
conda activate td-env
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Afterwards, training is done using:
```
python train.py
```
To specify a model id, use `--model_name name[version]`. A short explanation of model options from the paper are:
% table with columns: model_name arg, description

| model_name arg | description | #train runs |
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

To download the Entity dataset visit this [google drive](https://drive.google.com/drive/folders/1ghgYcEoCsbtSoQlMfD7-zUahYraU01zn) and unzip each of the three files in `data/entity/`. Additionally, download `entityseg_train_lr.json` and `entityseg_val_lr.json` from the official [github repo](https://github.com/adobe-research/EntitySeg-Dataset/releases/tag/v1.0) and place both files in `data/entity/`. Then run:

```bash
python data/entity/process_entity.py
```

This generates `data/entity/val_ims.txt`, `data/entity/test_ims.txt`, and per-folder mask images in `data/entity/entity_*_masks/*_mask.png`.