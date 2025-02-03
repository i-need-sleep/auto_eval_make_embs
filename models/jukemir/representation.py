import pathlib
from argparse import ArgumentParser

# imports and set up Jukebox's multi-GPU parallelization
import jukebox
from jukebox.hparams import Hyperparams, setup_hparams
from jukebox.make_models import MODELS, make_prior, make_vqvae
from jukebox.utils.dist_utils import setup_dist_from_mpi
from tqdm import tqdm

import librosa as lr
import numpy as np
import torch
import os

# --- MODEL PARAMS ---

JUKEBOX_SAMPLE_RATE = 44100
T = 8192
SAMPLE_LENGTH = 1048576  # ~23.77s, which is the param in JukeMIR
DEPTH = 36


# --------------------


def load_audio_from_file(fpath):
    audio, _ = lr.load(fpath, sr=JUKEBOX_SAMPLE_RATE)
    if audio.ndim == 1:
        audio = audio[np.newaxis]
    audio = audio.mean(axis=0)

    # normalize audio
    norm_factor = np.abs(audio).max()
    if norm_factor > 0:
        audio /= norm_factor

    return audio.flatten()


def audio_padding(audio, target_length):
    padding_length = target_length - audio.shape[0]
    padding_vector = np.zeros(padding_length)
    padded_audio = np.concatenate([audio, padding_vector], axis=0)
    return padded_audio


def get_z(audio, vqvae):
    # don't compute unnecessary discrete encodings
    audio = audio[: JUKEBOX_SAMPLE_RATE * 25]

    zs = vqvae.encode(torch.cuda.FloatTensor(audio[np.newaxis, :, np.newaxis]))

    z = zs[-1].flatten()[np.newaxis, :]

    if z.shape[-1] < 8192:
        raise ValueError("Audio file is not long enough")

    return z


def get_cond(hps, top_prior):
    sample_length_in_seconds = 62

    hps.sample_length = (
                                int(sample_length_in_seconds * hps.sr) // top_prior.raw_to_tokens
                        ) * top_prior.raw_to_tokens

    # NOTE: the 'lyrics' parameter is required, which is why it is included,
    # but it doesn't actually change anything about the `x_cond`, `y_cond`,
    # nor the `prime` variables
    metas = [
                dict(
                    artist="unknown",
                    genre="unknown",
                    total_length=hps.sample_length,
                    offset=0,
                    lyrics="""lyrics go here!!!""",
                ),
            ] * hps.n_samples

    labels = [None, None, top_prior.labeller.get_batch_labels(metas, "cuda")]

    x_cond, y_cond, prime = top_prior.get_cond(None, top_prior.get_y(labels[-1], 0))

    x_cond = x_cond[0, :T][np.newaxis, ...]
    y_cond = y_cond[0][np.newaxis, ...]

    return x_cond, y_cond


def get_final_activations(z, x_cond, y_cond, top_prior, return_hidden_states):
    x = z[:, :T]

    # make sure that we get the activations
    top_prior.prior.only_encode = True

    # encoder_kv and fp16 are set to the defaults, but explicitly so
    x, hidden_states = top_prior.prior.forward(
        x, x_cond=x_cond, y_cond=y_cond, encoder_kv=None, fp16=False, return_hidden_states=return_hidden_states
    )
    
    # hidden_states: List of (seq_len, emb_size)
    hidden_states = torch.cat([h for h in hidden_states], dim=0) # shape: (n_layers, seq_len, emb_size)

    # Record the different aggregations along the time axis
    first = hidden_states[:, 0, :].unsqueeze(1)
    last = hidden_states[:, -1, :].unsqueeze(1)
    average = hidden_states.mean(dim=1).unsqueeze(1)
    max_hidden_states = hidden_states.max(dim=1).values.unsqueeze(1)
    out = torch.cat([first, last, average, max_hidden_states], dim=1)

    return out


def get_acts_from_file(fpath, hps, vqvae, top_prior, meanpool, return_encodec=False):
    audio = load_audio_from_file(fpath)

    # zero padding
    if audio.shape[0] < SAMPLE_LENGTH:
        audio = audio_padding(audio, SAMPLE_LENGTH)
    
    # Truncation
    audio = audio[:SAMPLE_LENGTH]

    # run vq-vae on the audio
    z = get_z(audio, vqvae)

    # get conditioning info
    x_cond, y_cond = get_cond(hps, top_prior)

    # get the activations from the LM
    acts = get_final_activations(z, x_cond, y_cond, top_prior, [6, 13, 20, 27, 34]).type(torch.float32).cpu()
    return acts

def main(input_dir, output_dir):
    # --- SETTINGS ---

    DEVICE = 'cuda'
    VQVAE_MODELPATH = "/home/willhuang/projects/music_mauve/pretrained/vqvae.pth.tar"
    PRIOR_MODELPATH = "/home/willhuang/projects/music_mauve/pretrained/prior_level_2.pth.tar"
    INPUT_DIR = input_dir
    OUTPUT_DIR = output_dir
    AVERAGE_SLICES = 8192  # For average pooling. "1" means average all frames.
    #  Since the output shape is 8192 * 4800, the params bust can divide 8192.
    USING_CACHED_FILE = True
    model = "5b"  # might not fit to other settings, e.g., "1b_lyrics" or "5b_lyrics"

    # --- SETTINGS ---
    input_dir = pathlib.Path(INPUT_DIR)
    output_dir = pathlib.Path(OUTPUT_DIR)
    input_paths = sorted(list(input_dir.iterdir()))
    # filter
    input_paths = list(filter(lambda x: x.name.endswith('.wav') or x.name.endswith('mp3'), input_paths))
    device = DEVICE
    # Set up VQVAE

    hps = Hyperparams()
    hps.sr = 44100
    hps.n_samples = 8
    hps.name = "samples"
    chunk_size = 32
    max_batch_size = 16
    hps.levels = 3
    hps.hop_fraction = [0.5, 0.5, 0.125]
    vqvae, *priors = MODELS[model]
    hps_1 = setup_hparams(vqvae, dict(sample_length=SAMPLE_LENGTH))
    hps_1.restore_vqvae = VQVAE_MODELPATH
    vqvae = make_vqvae(
        hps_1, device
    )

    # Set up language model
    hps_2 = setup_hparams(priors[-1], dict())
    hps_2["prior_depth"] = DEPTH
    hps_2.restore_prior = PRIOR_MODELPATH
    top_prior = make_prior(hps_2, vqvae, device)

    for input_path in tqdm(input_paths):
        # Check if output already exists
        output_path = pathlib.Path(output_dir, f"{input_path.stem}.npy")
        if 'seed_3' in str(input_path):
            output_path = pathlib.Path(output_dir, f"{input_path.stem}_seed_3.npy")

        if os.path.exists(str(output_path)) and USING_CACHED_FILE:  # load cached data, and skip calculating
            np.load(output_path)
            continue

        # Decode, resample, convert to mono, and normalize audio
        try:
            with torch.no_grad():
                representation = get_acts_from_file(
                    input_path, hps, vqvae, top_prior, meanpool=AVERAGE_SLICES
                )
            
            np.save(output_path, representation)
        except Exception as e:
            print(f"Error processing {input_path}: {e}")


if __name__ == "__main__":
    output_root = '/mnt/sda/willhuang/music_eval_storage/data/embeddings/jukebox_5b'
    folder_root_a = '/data/sony/samples/musicgen'
    folder_root_b = '/data/sony/samples'
    folder_roots = [folder_root_a, folder_root_b]
    folders = []
    # for folder_root in folder_roots:
    #     for folder in os.listdir(folder_root):
    #         if not os.path.isdir(os.path.join(folder_root, folder, 'seed_3')):
    #             continue
    #         folders.append(os.path.join(folder_root, folder, 'seed_3'))
    folders.append('/data/sony/samples/audioldm2/seed_1/seed_1')
    folders.append('/data/sony/samples/musicldm/seed_1/seed_1')

    for folder in sorted(folders, reverse=False):
        input_dir = folder
        output_dir = os.path.join(output_root, folder.replace('/seed_1', '').split('/')[-1])
        # If the output directory exists, skip
        # if os.path.exists(output_dir):
        #     continue
        os.makedirs(output_dir, exist_ok=True)
        print(f"Processing {input_dir}")
        main(input_dir, output_dir)
        