from pathlib import Path 
import logging

import torch
import torch.nn.functional as F
import numpy as np
from hypy_utils.downloader import download_file
import librosa
from tqdm import tqdm

log = logging.getLogger(__name__)

class MusicGenSmall():
    def __init__(self, device, args):
        from models.modeling_musicgen import MusicgenForConditionalGeneration
        from transformers import MusicgenProcessor
        self.device = device
        self.sr = 32000

        # self.feature_extractor = MusicgenProcessor.from_pretrained(args['uglobals']['MUSICGEN_SMALL_PROCESSOR_DIR'])
        # self.model = MusicgenForConditionalGeneration.from_pretrained(args['uglobals']['MUSICGEN_SMALL_DIR'])
        self.feature_extractor = MusicgenProcessor.from_pretrained('facebook/musicgen-small')
        self.model = MusicgenForConditionalGeneration.from_pretrained('facebook/musicgen-small')
        self.model.to(device)
        self.model.eval()
    
    def make_input_ids(self, audio_input):
        # Unconditional text embeddings
        inputs_tensor, model_input_name, model_kwargs = self.model._prepare_model_inputs(
            None, self.model.generation_config.bos_token_id, audio_input
        )

        model_kwargs = self.model._prepare_text_encoder_kwargs_for_generation(
            inputs_tensor,
            model_kwargs,
            model_input_name,
            guidance_scale=self.model.generation_config.guidance_scale,
        )

        # Encode audio
        model_kwargs = self.model._prepare_audio_encoder_kwargs_for_generation(
            model_kwargs["input_values"].to(self.device),
            model_kwargs,
        )

        input_ids, model_kwargs = self.model._prepare_decoder_input_ids_for_generation(
            batch_size=inputs_tensor.shape[0],
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=self.model.generation_config.decoder_start_token_id,
            bos_token_id=self.model.generation_config.bos_token_id,
            device=inputs_tensor.device,
        )
        input_ids = input_ids[: , 1:] # Slice off the BoS token
        model_kwargs['encoder_outputs'].last_hidden_state = torch.zeros_like(model_kwargs['encoder_outputs'].last_hidden_state) # To be consistent with the unconditional inputs
        return input_ids, model_kwargs['encoder_outputs']
    
    def add_bos(self, ids):
        return torch.cat([torch.ones(ids.shape[0], 1).long().to(self.device) * 2048, ids], dim=1)

    def make_delay_pattern(self, input_ids):
        # Add BoS
        input_ids = self.add_bos(input_ids)
        input_ids, decoder_delay_pattern_mask = self.model.decoder.build_delay_pattern_mask(
            input_ids,
            pad_token_id=2048, # Ugly
            max_length=1505,
        )
        return input_ids, decoder_delay_pattern_mask

    def get_embedding(self, audio):
        with torch.no_grad():
            input_features = self.feature_extractor(audio, sampling_rate=self.sr, return_tensors="pt").to(self.device)
            audio_input_ids, audio_encoder_outputs = self.make_input_ids(input_features)
            audio_input_ids, decoder_delay_pattern_mask = self.make_delay_pattern(audio_input_ids) # With BoS

            hidden_states = self.model.forward_with_loss_mask(encoder_outputs=audio_encoder_outputs, labels=audio_input_ids[:, 1:], loss_mask=torch.zeros_like(audio_input_ids[:, 1:]), output_hidden_states=True).decoder_hidden_states
            hidden_states = torch.stack(hidden_states, dim=0)
            # [#layers, batch, timeframes, 1024]

            # Take the 6, 12, 18, 24th payers
            hidden_states = hidden_states[[6, 12, 18, 14]]
            # [4, batch, timeframes, 1024]

            first = hidden_states[:, :, 0, :].unsqueeze(2)
            last = hidden_states[:, :, -1, :].unsqueeze(2)
            average = hidden_states.mean(dim=2).unsqueeze(2)
            max_hidden_states = hidden_states.max(dim=2).values.unsqueeze(2)
            hidden_states = torch.cat([first, last, average, max_hidden_states], dim=2)
        return hidden_states 
    
class MusicGenMedium():
    def __init__(self, device, args):
        from models.modeling_musicgen import MusicgenForConditionalGeneration
        from transformers import MusicgenProcessor
        self.device = device
        self.sr = 32000

        # self.feature_extractor = MusicgenProcessor.from_pretrained(args['uglobals']['MUSICGEN_MEDIUM_PROCESSOR_DIR'])
        # self.model = MusicgenForConditionalGeneration.from_pretrained(args['uglobals']['MUSICGEN_MEDIUM_DIR'])
        self.feature_extractor = MusicgenProcessor.from_pretrained('facebook/musicgen-medium')
        self.model = MusicgenForConditionalGeneration.from_pretrained('facebook/musicgen-medium')


        self.model.to(device)
        self.model.eval()
    
    def make_input_ids(self, audio_input):
        # Unconditional text embeddings
        inputs_tensor, model_input_name, model_kwargs = self.model._prepare_model_inputs(
            None, self.model.generation_config.bos_token_id, audio_input
        )

        model_kwargs = self.model._prepare_text_encoder_kwargs_for_generation(
            inputs_tensor,
            model_kwargs,
            model_input_name,
            guidance_scale=self.model.generation_config.guidance_scale,
        )

        # Encode audio
        model_kwargs = self.model._prepare_audio_encoder_kwargs_for_generation(
            model_kwargs["input_values"].to(self.device),
            model_kwargs,
        )

        input_ids, model_kwargs = self.model._prepare_decoder_input_ids_for_generation(
            batch_size=inputs_tensor.shape[0],
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=self.model.generation_config.decoder_start_token_id,
            bos_token_id=self.model.generation_config.bos_token_id,
            device=inputs_tensor.device,
        )
        input_ids = input_ids[: , 1:] # Slice off the BoS token
        model_kwargs['encoder_outputs'].last_hidden_state = torch.zeros_like(model_kwargs['encoder_outputs'].last_hidden_state) # To be consistent with the unconditional inputs
        return input_ids, model_kwargs['encoder_outputs']
    
    def add_bos(self, ids):
        return torch.cat([torch.ones(ids.shape[0], 1).long().to(self.device) * 2048, ids], dim=1)

    def make_delay_pattern(self, input_ids):
        # Add BoS
        input_ids = self.add_bos(input_ids)
        input_ids, decoder_delay_pattern_mask = self.model.decoder.build_delay_pattern_mask(
            input_ids,
            pad_token_id=2048, # Ugly
            max_length=1505,
        )
        return input_ids, decoder_delay_pattern_mask    
    
    def get_embedding(self, audio):
        with torch.no_grad():
            input_features = self.feature_extractor(audio, sampling_rate=self.sr, return_tensors="pt").to(self.device)
            audio_input_ids, audio_encoder_outputs = self.make_input_ids(input_features)
            audio_input_ids, decoder_delay_pattern_mask = self.make_delay_pattern(audio_input_ids) # With BoS

            hidden_states = self.model.forward_with_loss_mask(encoder_outputs=audio_encoder_outputs, labels=audio_input_ids[:, 1:], loss_mask=torch.zeros_like(audio_input_ids[:, 1:]), output_hidden_states=True).decoder_hidden_states
            hidden_states = torch.stack(hidden_states, dim=0)
            # [#layers, batch, timeframes, 1024]

            # Take the every 6th payer
            hidden_states = hidden_states[[i for i in range(0, len(self.model.decoder.model.decoder.layers), 6)]]
            # [4, batch, timeframes, 1024]

            first = hidden_states[:, :, 0, :].unsqueeze(2)
            last = hidden_states[:, :, -1, :].unsqueeze(2)
            average = hidden_states.mean(dim=2).unsqueeze(2)
            max_hidden_states = hidden_states.max(dim=2).values.unsqueeze(2)
            hidden_states = torch.cat([first, last, average, max_hidden_states], dim=2)

        return hidden_states # [4, batch, 4, 1024]
    
class MusicGenMediumEncodecOnly():
    def __init__(self, device, args):
        from models.modeling_musicgen import MusicgenForConditionalGeneration
        from transformers import MusicgenProcessor
        self.device = device
        self.sr = 32000

        self.feature_extractor = MusicgenProcessor.from_pretrained(args['uglobals']['MUSICGEN_MEDIUM_PROCESSOR_DIR'])
        self.model = MusicgenForConditionalGeneration.from_pretrained(args['uglobals']['MUSICGEN_MEDIUM_DIR'])
        self.model.to(device)
        self.model.eval()
    
    def make_input_ids(self, audio_input):
        # Unconditional text embeddings
        inputs_tensor, model_input_name, model_kwargs = self.model._prepare_model_inputs(
            None, self.model.generation_config.bos_token_id, audio_input
        )

        model_kwargs = self.model._prepare_text_encoder_kwargs_for_generation(
            inputs_tensor,
            model_kwargs,
            model_input_name,
            guidance_scale=self.model.generation_config.guidance_scale,
        )

        # Encode audio
        model_kwargs = self.model._prepare_audio_encoder_kwargs_for_generation(
            model_kwargs["input_values"].to(self.device),
            model_kwargs,
        )

        input_ids, model_kwargs = self.model._prepare_decoder_input_ids_for_generation(
            batch_size=inputs_tensor.shape[0],
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=self.model.generation_config.decoder_start_token_id,
            bos_token_id=self.model.generation_config.bos_token_id,
            device=inputs_tensor.device,
        )
        input_ids = input_ids[: , 1:] # Slice off the BoS token
        model_kwargs['encoder_outputs'].last_hidden_state = torch.zeros_like(model_kwargs['encoder_outputs'].last_hidden_state) # To be consistent with the unconditional inputs
        return input_ids, model_kwargs['encoder_outputs']
    
    def add_bos(self, ids):
        return torch.cat([torch.ones(ids.shape[0], 1).long().to(self.device) * 2048, ids], dim=1)

    def make_delay_pattern(self, input_ids):
        # Add BoS
        input_ids = self.add_bos(input_ids)
        input_ids, decoder_delay_pattern_mask = self.model.decoder.build_delay_pattern_mask(
            input_ids,
            pad_token_id=2048, # Ugly
            max_length=1505,
        )
        return input_ids, decoder_delay_pattern_mask    
    
    def get_embedding(self, audio):
        with torch.no_grad():
            input_features = self.feature_extractor(audio, sampling_rate=self.sr, return_tensors="pt").to(self.device)
            audio_input_ids, audio_encoder_outputs = self.make_input_ids(input_features)
            audio_input_ids, decoder_delay_pattern_mask = self.make_delay_pattern(audio_input_ids) # With BoS
        return audio_input_ids.reshape(1, audio_input_ids.shape[0], audio_input_ids.shape[1], 1)
    
class MusicGenLarge():
    def __init__(self, device, args):
        from models.modeling_musicgen import MusicgenForConditionalGeneration
        from transformers import MusicgenProcessor
        self.device = device
        self.sr = 32000

        # self.feature_extractor = MusicgenProcessor.from_pretrained(args['uglobals']['MUSICGEN_LARGE_PROCESSOR_DIR'])
        # self.model = MusicgenForConditionalGeneration.from_pretrained(args['uglobals']['MUSICGEN_LARGE_DIR'])
        self.feature_extractor = MusicgenProcessor.from_pretrained('facebook/musicgen-large')
        self.model = MusicgenForConditionalGeneration.from_pretrained('facebook/musicgen-large')
        self.model.to(device)
        self.model.eval()
    
    def make_input_ids(self, audio_input):
        # Unconditional text embeddings
        inputs_tensor, model_input_name, model_kwargs = self.model._prepare_model_inputs(
            None, self.model.generation_config.bos_token_id, audio_input
        )

        model_kwargs = self.model._prepare_text_encoder_kwargs_for_generation(
            inputs_tensor,
            model_kwargs,
            model_input_name,
            guidance_scale=self.model.generation_config.guidance_scale,
        )

        # Encode audio
        model_kwargs = self.model._prepare_audio_encoder_kwargs_for_generation(
            model_kwargs["input_values"].to(self.device),
            model_kwargs,
        )

        input_ids, model_kwargs = self.model._prepare_decoder_input_ids_for_generation(
            batch_size=inputs_tensor.shape[0],
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=self.model.generation_config.decoder_start_token_id,
            bos_token_id=self.model.generation_config.bos_token_id,
            device=inputs_tensor.device,
        )
        input_ids = input_ids[: , 1:] # Slice off the BoS token
        model_kwargs['encoder_outputs'].last_hidden_state = torch.zeros_like(model_kwargs['encoder_outputs'].last_hidden_state) # To be consistent with the unconditional inputs
        return input_ids, model_kwargs['encoder_outputs']
    
    def add_bos(self, ids):
        return torch.cat([torch.ones(ids.shape[0], 1).long().to(self.device) * 2048, ids], dim=1)

    def make_delay_pattern(self, input_ids):
        # Add BoS
        input_ids = self.add_bos(input_ids)
        input_ids, decoder_delay_pattern_mask = self.model.decoder.build_delay_pattern_mask(
            input_ids,
            pad_token_id=2048, # Ugly
            max_length=1505,
        )
        return input_ids, decoder_delay_pattern_mask

    def get_embedding(self, audio):
        with torch.no_grad():
            input_features = self.feature_extractor(audio, sampling_rate=self.sr, return_tensors="pt").to(self.device)
            audio_input_ids, audio_encoder_outputs = self.make_input_ids(input_features)
            audio_input_ids, decoder_delay_pattern_mask = self.make_delay_pattern(audio_input_ids) # With BoS

            hidden_states = self.model.forward_with_loss_mask(encoder_outputs=audio_encoder_outputs, labels=audio_input_ids[:, 1:], loss_mask=torch.zeros_like(audio_input_ids[:, 1:]), output_hidden_states=True).decoder_hidden_states
            hidden_states = torch.stack(hidden_states, dim=0)
            # [#layers, batch, timeframes, 1024]

            # Take the every 6th payer
            hidden_states = hidden_states[[i for i in range(0, len(self.model.decoder.model.decoder.layers), 6)]]
            # [4, batch, timeframes, 1024]

            first = hidden_states[:, :, 0, :].unsqueeze(2)
            last = hidden_states[:, :, -1, :].unsqueeze(2)
            average = hidden_states.mean(dim=2).unsqueeze(2)
            max_hidden_states = hidden_states.max(dim=2).values.unsqueeze(2)
            hidden_states = torch.cat([first, last, average, max_hidden_states], dim=2)

        return hidden_states 
    
class MERT330M():
    def __init__(self, device):
        from transformers import Wav2Vec2FeatureExtractor
        from transformers import AutoModel

        self.sr = 24000
        self.device = device
        
        self.model = AutoModel.from_pretrained('m-a-p/MERT-v1-330M', trust_remote_code=True)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained('m-a-p/MERT-v1-330M', trust_remote_code=True)
        self.model.eval()
        self.model.to(self.device)

    def get_embedding(self, audio):
        inputs = self.processor(audio, sampling_rate=self.sr, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
            hidden_states = torch.stack(out.hidden_states)

            # Take the 6, 12, 18, 24th payers
            hidden_states = hidden_states[[6, 12, 18, 14]]
            # [4, batch, timeframes, 1024]

            first = hidden_states[:, :, 0, :].unsqueeze(2)
            last = hidden_states[:, :, -1, :].unsqueeze(2)
            average = hidden_states.mean(dim=2).unsqueeze(2)
            max_hidden_states = hidden_states.max(dim=2).values.unsqueeze(2)
            hidden_states = torch.cat([first, last, average, max_hidden_states], dim=2)
        return hidden_states
    
class CLAPLaionAud():
    """
    CLAP model from https://github.com/LAION-AI/CLAP
    """
    
    def __init__(self, device):
        self.sr = 48000
        self.device = device

        url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-best.pt'
        self.model_file = Path(__file__).parent / ".model-checkpoints" / url.split('/')[-1]

        # Download file if it doesn't exist
        if not self.model_file.exists():
            self.model_file.parent.mkdir(parents=True, exist_ok=True)
            download_file(url, self.model_file)
            
        # Patch the model file to remove position_ids (will raise an error otherwise)
        self.patch_model_430(self.model_file)
        self.load_model()

    def patch_model_430(self, file):
        """
        Patch the model file to remove position_ids (will raise an error otherwise)
        This is a new issue after the transformers 4.30.0 update
        Please refer to https://github.com/LAION-AI/CLAP/issues/127
        """
        # Create a "patched" file when patching is done
        patched = file.parent / f"{file.name}.patched.430"
        if patched.exists():
            return
        
        OFFENDING_KEY = "module.text_branch.embeddings.position_ids"
        log.warning("Patching LAION-CLAP's model checkpoints")
        
        # Load the checkpoint from the given path
        checkpoint = torch.load(file, map_location="cpu")

        # Extract the state_dict from the checkpoint
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Delete the specific key from the state_dict
        if OFFENDING_KEY in state_dict:
            del state_dict[OFFENDING_KEY]

        # Save the modified state_dict back to the checkpoint
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint["state_dict"] = state_dict

        # Save the modified checkpoint
        torch.save(checkpoint, file)
        log.warning(f"Saved patched checkpoint to {file}")
        
        # Create a "patched" file when patching is done
        patched.touch()
        
    def load_model(self):
        import laion_clap

        self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')
        self.model.load_ckpt(self.model_file)
        self.model.eval()
        self.model.to(self.device)

    def get_embedding(self, audio):
        # Audio: List of np.array
        # Zero pad the audio to have the same length
        max_len = max([len(a) for a in audio])
        audio = [np.pad(a, (0, max_len - len(a))) for a in audio]
        audio = torch.tensor(audio).numpy()
        
        # The int16-float32 conversion is used for quantization
        audio = self.int16_to_float32(self.float32_to_int16(audio))

        # Split the audio into 10s chunks with 1s hop
        chunk_size = 10 * self.sr  # 10 seconds
        hop_size = self.sr  # 1 second
        chunks = [audio[:, i:i+chunk_size] for i in range(0, audio.shape[1], hop_size)]

        # Calculate embeddings for each chunk
        embeddings = []
        for chunk in chunks:
            with torch.no_grad():
                chunk = chunk if chunk.shape[1] == chunk_size else np.pad(chunk, ((0,0), (0, chunk_size-chunk.shape[1])))
                chunk = torch.from_numpy(chunk).float().to(self.device)
                emb = self.model.get_audio_embedding_from_data(x = chunk, use_tensor=True).unsqueeze(1)
                embeddings.append(emb)

        # Concatenate the embeddings
        emb = torch.cat(embeddings, dim=1) # [batch_size, time_frames, 512]
        
        # Take the average along the time dimension
        mean = emb.mean(dim=1).unsqueeze(1)
        max_emb = emb.max(dim=1).values.unsqueeze(1)
        emb = torch.cat([mean, max_emb], dim=1).unsqueeze(0)
        return emb

    def int16_to_float32(self, x):
        return (x / 32767.0).astype(np.float32)

    def float32_to_int16(self, x):
        x = np.clip(x, a_min=-1., a_max=1.)
        return (x * 32767.).astype(np.int16)

class JohnMusicGenSmall():
    def __init__(self, device, args, vocab):

        self.device = device
        self.vocab = vocab
        self.sr = 48000 # Read the audio at 48kHz, reconstruct with Encodec, then encode with the 32K Encodec

        import sys
        # Use the hacked version of the transformers for MusicGen
        sys.path.insert(0, '/home/willhuang/projects/music_mauve/code/models/transformers_levanter_audio_ape/src')
        import transformers as transformers_moded

        self.model = transformers_moded.AutoModelForCausalLM.from_pretrained(args['uglobals']['JOHN_MUSICGEN_SMALL_DIR']).to(device)
        self.model.eval()

        # Clear the modified transformers from sys.modules
        if 'transformers' in sys.modules:
            del sys.modules['transformers']
        # Also remove related modules that may be cached from the modified package
        modules_to_clear = [key for key in sys.modules if key.startswith('transformers.')]
        for module in modules_to_clear:
            del sys.modules[module]

        sys.path = sys.path[1:]
        import transformers

        self.encodec_48k = transformers.EncodecModel.from_pretrained("facebook/encodec_48khz").to(device)
        self.encodec_48k_processor = transformers.AutoProcessor.from_pretrained("facebook/encodec_48khz")
        self.encodec_48k.eval()

        self.encodec_32k = transformers.EncodecModel.from_pretrained("facebook/encodec_32khz").to(device)
        self.encodec_32k_processor = transformers.AutoProcessor.from_pretrained("facebook/encodec_32khz")
        self.encodec_32k.eval()
    
    def skew(self, blocks, block_size, pad):
        # MusicGen-style interleaving
        codes = F.pad(blocks, (0,block_size-1), mode='constant', value=pad)
        codes = torch.stack([torch.roll(codes[i], i) for i in range(block_size)])

        # flatten the codes into a sequence
        return codes.T.flatten()
    
    def deskew(self, tokens, block_size):
        # unroll the MusicGen interleaving
        blocks = torch.tensor(tokens).reshape(-1, block_size).T
        blocks = torch.stack([torch.roll(blocks[i], -i) for i in range(block_size)])[:,:-(block_size-1)]

        return blocks
    
    def prep_tokens(self, tokens):
        out = torch.zeros(tokens.shape[0], 4, tokens.shape[2] + 3, dtype=torch.long).to(self.device) - 1
        for i in range(4):
            out[:, 3 - i, i: i - 3 + out.shape[-1]] = tokens[:, i, :]
        return out
    
    def unprep_tokens(self, tokens):
        out = torch.zeros(tokens.shape[0], 4, tokens.shape[2] - 3, dtype=torch.long).to(self.device)
        for i in range(4):
            out[:, i, :] = tokens[:, 3 - i, i: i - 3 + tokens.shape[-1]]
        return out
    
    def tokenize(self, frames, scales, vocab):
        residuals = vocab['config']['residuals']
        offsets = torch.tensor(vocab['residual_offset'])[:,None].to(self.device)
        assert residuals > 0

        # truncate unused residuals and add offsets for each residual vocabulary
        frames = [frame[0,:residuals] + offsets for frame in frames]

        # represent scales with dummy residuals so that the model can treat everything homogeneously
        scales = [torch.tensor([s + vocab['scale_offset']] + (residuals-1)*[vocab['scale_pad']]).view(residuals,1) for s in scales]

        # tack the scales onto the front of each (1-second) block of audio codes
        chunks = [v for pair in zip(scales, frames) for v in pair]

        return torch.cat(chunks, axis=1)
    
    def detokenize(self, blocks, vocab):
        residuals = vocab['config']['residuals']
        offsets = torch.tensor(vocab['residual_offset'])[:,None]
        assert residuals > 0

        # remove offsets for the residual vocabularies
        for i in range(residuals):
            blocks[i] = blocks[i] - offsets[i]

        return blocks.view(1,1,4,-1)

    def safe_audio(self, logits):
        sep = self.vocab['separator']
        pad = self.vocab['residual_pad']
        control_offset = self.vocab['control_offset']
        residual_offsets = self.vocab['residual_offset']
        codebook_size = self.vocab['config']['codebook_size']

        logits[0, pad] = -float('inf') # don't terminate
        for i in range(4):
            logits[i,control_offset:residual_offsets[i]] = -float('inf')
            logits[i,residual_offsets[i]+codebook_size:] = -float('inf')

        return logits

    def get_embedding(self, audio):
        # Audio: List of np.array
        # Zero pad the audio to have the same length
        max_len = max([len(a) for a in audio])
        audio = [np.pad(a, (0, max_len - len(a))).reshape(1, -1) for a in audio] 
        # [(1, n_samples)]
        # Duplicate each array into stereo
        audio = [np.concatenate([a, a], axis=0) for a in audio]
        # [(2, n_samples)]

        with torch.no_grad():
            # Encode the audio at 48kHz
            inputs = self.encodec_48k_processor(raw_audio=audio, sampling_rate=self.encodec_48k_processor.sampling_rate, return_tensors="pt")
            audio_values = self.encodec_48k(inputs["input_values"].to(self.device)).audio_values.cpu().numpy()
            
            # Re-encode the audio at 32kHz
            # Porcess the audio into [(samples, )]
            audio_values = [librosa.resample(a[0], orig_sr=48000, target_sr=32000) for a in audio_values]
            inputs = self.encodec_32k_processor(raw_audio=audio_values, sampling_rate=self.encodec_32k_processor.sampling_rate, return_tensors="pt")
            encoded = self.encodec_32k.encode(inputs["input_values"].to(self.device))
            encoded = encoded['audio_codes'][0]
            # [batch_size, 4, seq_len]

            # Apply offsets
            offsets = torch.tensor(self.vocab['residual_offset'])[:,None].to(self.device)
            for i in range(4):
                encoded[:, i] = encoded[:, i] + offsets[i]
            encoded = encoded[:, :, :1000]

            # Prepare inputs for the LM
            # Prepare a blank prompt
            # This might need to be changed depending on the model
            prompt = torch.tensor([3, 7, 2, 2]).to(self.device)
            prompt = prompt.unsqueeze(0).repeat(encoded.shape[0], 1)
            
            # Skew and concatenate the prompt and the audio embeddings
            skewed = torch.stack([self.skew(e, 4, 0) for e in encoded])[:, 12: -12]
            skewed = torch.cat([prompt, skewed], dim=1)
            
            hidden_states = self.model(skewed, output_hidden_states=True).hidden_states

            hidden_states = torch.stack(hidden_states, dim=0)
            # [#layers, batch, timeframes, 1024]

            # Take the 6th and 12th payers
            hidden_states = hidden_states[[6, 12]]
            # [4, batch, timeframes, 1024]

            first = hidden_states[:, :, 0, :].unsqueeze(2)
            last = hidden_states[:, :, -1, :].unsqueeze(2)
            average = hidden_states.mean(dim=2).unsqueeze(2)
            max_hidden_states = hidden_states.max(dim=2).values.unsqueeze(2)
            hidden_states = torch.cat([first, last, average, max_hidden_states], dim=2)
            
        return hidden_states 

            # for _ in tqdm(range(250)):
            #     # Unit test: Generate the embeddings
            #     # outputs = self.model(torch.tensor([[3, 7, 2, 2]], device='cuda:0'))
            #     logits = outputs.logits[0, -4:,:]
            #     logits = self.safe_audio(logits)
            #     probabilities = torch.softmax(logits, dim=-1).double()
                
            #     next_token = torch.multinomial(probabilities, num_samples=1).to(prompt.device).squeeze().tolist()
            #     next_token = torch.tensor(next_token).unsqueeze(0).to(self.device)
            #     skewed = torch.cat([skewed, next_token], dim=1)

            # # Strip the context tokens
            # skewed = skewed[0]
            # skewed = skewed[4:]

            # # strip sequence separators
            # tokens = [token for token in skewed if token != self.vocab['separator']]
            # blocks = self.deskew(tokens, 4)
            
            # encoded = self.detokenize(blocks, self.vocab).to(self.device)
            # audio_values = self.encodec_32k.decode(encoded.int(), [None])[0]
            
            # from scipy.io.wavfile import write
            # write('test.wav', 32000, audio_values.cpu().numpy())
            # exit()

    
class VGGish():
    def __init__(self, device):

        self.sr = 16000
        self.device = device
        
        self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.model.postprocess = False
        self.model.embeddings = torch.nn.Sequential(*list(self.model.embeddings.children())[:-1])
        self.model.eval()
        self.model.to(self.device)

    def get_embedding(self, audios):
        with torch.no_grad():
            out = []
            for audio in audios:
                out.append(self.model.forward(audio, self.sr))
            out = torch.stack(out)
            # [batch, timeframes, 128]

            mean = out.mean(dim=1).unsqueeze(1).unsqueeze(0)
            hidden_states = mean.repeat(4, 1, 4, 1)
        return hidden_states