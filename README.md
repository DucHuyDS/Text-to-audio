
## Download checkpoints and dataset
1. Download checkpoints from Google Drive: [link](https://drive.google.com/file/d/1T6EnuAHIc8ioeZ9kB1OZ_WGgwXAVGOZS/view?usp=drive_link). The checkpoints including pretrained VAE, AudioMAE, CLAP, 16kHz HiFiGAN, and 48kHz HiFiGAN.
2. Uncompress the checkpoint tar file and place the content into **data/checkpoints/**



# Play around with the code

## Train the AudioLDM model
```python
# Train the AudioLDM (latent diffusion part)
python3 audioldm_train/train/latent_diffusion.py -c audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original.yaml

# Train the VAE (Optional)
# python3 audioldm_train/train/autoencoder.py -c audioldm_train/config/2023_11_13_vae_autoencoder/16k_64.yaml
```

The program will perform generation on the evaluation set every 5 epochs of training. After obtaining the audio generation folders (named val_<training-steps>), you can proceed to the next step for model evaluation.

## Finetuning of the pretrained model

You can finetune with two pretrained checkpoint, first download the one that you like (e.g., using wget):
1. Medium size AudioLDM: https://zenodo.org/records/7884686/files/audioldm-m-full.ckpt
2. Small size AudioLDM: https://zenodo.org/records/7884686/files/audioldm-s-full

Place the checkpoint in the *data/checkpoints* folder

Then perform finetuning with one of the following commands:
```shell
# Medium size AudioLDM
python3 audioldm_train/train/latent_diffusion.py -c audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml --reload_from_ckpt data/checkpoints/audioldm-m-full.ckpt

# Small size AudioLDM
python3 audioldm_train/train/latent_diffusion.py -c audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original.yaml --reload_from_ckpt data/checkpoints/audioldm-s-full
```
You can specify your own dataset following the same format as the provided AudioCaps dataset.

Note that the pretrained AudioLDM checkpoints are under CC-by-NC 4.0 license, which is not allowed for commerial use.

## Evaluate the model output
Automatically evaluation based on each of the folder with generated audio
```python

# Evaluate all existing generated folder
python3 audioldm_train/eval.py --log_path all

# Evaluate only a specific experiment folder
python3 audioldm_train/eval.py --log_path <path-to-the-experiment-folder>
```
The evaluation result will be saved in a json file at the same level of the audio folder.

## Inference with the pretrained model
Use the following syntax:

```shell
python3 audioldm_train/infer.py --config_yaml <The-path-to-the-same-config-file-you-use-for-training> --list_inference <the-filelist-you-want-to-generate>
```

For example:
```shell
# Please make sure you have train the model using audioldm_crossattn_flant5.yaml
# The generated audio will be saved at the same log folder if the pretrained model.
python3 audioldm_train/infer.py --config_yaml audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original.yaml --list_inference tests/captionlist/test.json
```



The generated audio will be named with the caption by default. If you like to specify the filename to use, please checkout the format of *tests/captionlist/inference_test_with_filename.lst*.

This repo only support inference with the model you trained by yourself. If you want to use the pretrained model directly, please use these two repos: [AudioLDM](https://github.com/haoheliu/AudioLDM) and [AudioLDM2](https://github.com/haoheliu/AudioLDM2).
