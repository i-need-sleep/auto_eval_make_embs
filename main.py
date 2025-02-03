import os
import argparse

import torch

from tqdm import tqdm
import librosa

import utils.logging_utils as logging_utils

import models.embedding_models as embedding_models

from models.jukemir import representation as jukemir_representation

def main(args):
    # Device
    if not torch.cuda.is_available() or args.force_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    print(f'Device: {device}')

    # Output dir
    output_dir = os.path.join(args.output_dir, args.model_type)
    os.makedirs(output_dir, exist_ok=True)
    
    # Model
    if args.model_type == 'musicgen_small':
        model = embedding_models.MusicGenSmall(device, vars(args))
    elif args.model_type == 'musicgen_medium':
        model = embedding_models.MusicGenMedium(device, vars(args))
    # elif args.model_type == 'musicgen_medium_encodec_only':
    #     model = embedding_models.MusicGenMediumEncodecOnly(device, vars(args))
    # elif args.model_type == 'musicgen_large':
    #     model = embedding_models.MusicGenLarge(device, vars(args))
    elif args.model_type == 'mert_330m':
        model = embedding_models.MERT330M(device)
    elif args.model_type == 'clap_l_aud':
        model = embedding_models.CLAPLaionAud(device)
    # elif args.model_type == 'john_musicgen_small':
    #     from anticipation_long_context.anticipation.vocabs.mmvocab import vocab 
    #     model = embedding_models.JohnMusicGenSmall(device, vars(args), vocab)
    elif args.model_type == 'vggish':
        model = embedding_models.VGGish(device)
    else:
        raise ValueError(f'Invalid model type: {args.model_type}')
    
    #Embed
    batch_files = []
    batch_audios = []
    for idx, file in tqdm(enumerate(os.listdir(args.input_dir)), total=len(os.listdir(args.input_dir))):
        if not file.endswith('.wav') and not file.endswith('.mp3'):
            continue
        # If the output file already exists, skip
        out_name = file.replace('.wav', '.pt').replace('.mp3', '.pt')
        if os.path.exists(os.path.join(output_dir, out_name)):
            continue

        file_path = os.path.join(args.input_dir, file)
        try:
            audio, _ = librosa.load(file_path, sr=model.sr)
        except:
            print(f'Error loading {file_path}')
            continue
        audio = audio[:int(model.sr * 29.8)] # Dirty fix
        batch_files.append(file)
        batch_audios.append(audio)
        if len(batch_audios) == args.batch_size or idx == len(os.listdir(args.input_dir)) - 1:
            try:
                batch_embs = model.get_embedding(batch_audios)
                for i, file in enumerate(batch_files):
                    batch_embs[:, i, :, :]
                    out_name = file.replace('.wav', '.pt').replace('.mp3', '.pt')
                    torch.save(batch_embs[:, i, :, :], os.path.join(output_dir, out_name))
            except:
                print(f'Error embedding {file_path}')
                
            batch_files = []
            batch_audios = []
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Debugging
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--force_cpu', action='store_true')

    # Model
    parser.add_argument('--model_type', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=3)

    # Embedding
    parser.add_argument('--input_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./embeddings')

    # Presets
    parser.add_argument('--all', action='store_true')
    
    args = parser.parse_args()
    
    if args.all:
        for model_type in ['vggish', 'musicgen_small', 'musicgen_medium', 'mert_330m', 'clap_l_aud']:
            args.model_type = model_type
            main(args)
        
        os.makedirs(f'{args.output_dir}/jukebox_5b', exist_ok=True)
        jukemir_representation.main(args.input_dir, f'{args.output_dir}/jukebox_5b')