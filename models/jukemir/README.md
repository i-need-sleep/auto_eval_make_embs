# Simplified JukeMIR: A minimum version for extracting features with Jukebox

This repository contains code is for extracting JukeMIR features in a simplified way on your server.

It generates a (N, 4800) numpy array for a ~23.77s audio clip, where N depends on the local pooling parameter. The script will automatically cut or pad the waveform to this length.

It requires a single GPU with 12GB memory.

## Change Logs

### v1.11

- I moved the Jukebox model loading code out of the loop, which made the script faster overall.

### v1.1

- I optimized the parameter settings. Now the model parameters of JukeMIR and the parameters we need to set manually are placed separately.
- I removed some useless parameters.
- I added a new parameter `AVERAGE_SLICES` for feature extraction, which allows us to do local average pooling instead of always global pooling.
- Some other optimizations.

### v1.0

- I remove files unrelated to the feature extraction, making the code that can be easily embedded into your own projects.
- I provide a guide that you can set up the environment without `docker` (since we usually do not have `sudo` permission in our servers).
- I fix the code file to make sure we can set up the pretrained models path manually.
- I remove the `mpi4py` dependency, which causes errors in the environment.
- I add a zero-padding code so music clips shorter than `23.44s` will be processed correctly.
- I fix the prior model loading error.
- etc.

## Installation Guide

### Step 1: Installing Jukebox

The simplified JukeMIR package is basically the same as the original version, but I removed the `mpi4py` module, which is for parallel computing. `mpi4py` is difficult to be installed to the environment and has no effect on feature extraction. 

Run:

```
python -m pip install --no-cache-dir -e jukebox
```

### Step 2: Downloading pretrained models

Jukebox consists of two models: the VQ-VAE model and the prior model.

Run:

```
wget https://openaipublic.azureedge.net/jukebox/models/5b/vqvae.pth.tar
wget https://openaipublic.azureedge.net/jukebox/models/5b/prior_level_2.pth.tar
```

You can move the pretrained models to anywhere you like.

### Step 3: Changing the parameters

I recommend to precalculate the representations of your data in advance, since it takes really long.

The parameters you may change are in `representation.py`:

- `DEVICE='cuda'`: It determines whether you run the code on the GPU or not.
- `VQVAE_MODELPATH = "models/5b/vqvae.pth.tar"`, `PRIOR_MODELPATH = "models/5b/prior_level_2.pth.tar"`,`INPUT_DIR = r"BUTTER_v2/westone-dataset/WAV/"`,`OUTPUT_DIR = r"BUTTER_v2/jukemir/output/"`: They specify where your pre-trained model, audio files, and representation outputs are stored.
-  `AVERAGE_SLICES = 32`: New parameter. Specifies how many chunks the representation of the model output is divided into and averaged. When it is specified as `1`, the code will globally average the output pooling. This parameter must be divisible by `8192`.
- `USING_CACHED_FILE = False`: New parameter. When it is set to True, the code will check if the output folder already exists for the output file corresponding to the current audio file, and if it does, skip it.
- `model = "5b"`: Jukebox is available in different sizes, with the default version of JukeMIR being `5b`. Some optional parameters are `5b_lyrics` and `1b`, but neither has been tested.

### Step 4: Run

Run:

```
python representation.py
```

and you will get the results.

## Citation

If you use this codebase in your work, please consider citing the original paper:

```
@inproceedings{castellon2021calm,
  title={Codified audio language modeling learns useful representations for music information retrieval},
  author={Castellon, Rodrigo and Donahue, Chris and Liang, Percy},
  booktitle={ISMIR},
  year={2021}
}
```

You are also welcome to add a footnote referring to this project:

```
https://github.com/ldzhangyx/simplified-jukemir
```
