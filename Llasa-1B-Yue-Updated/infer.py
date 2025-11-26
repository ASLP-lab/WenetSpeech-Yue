import argparse

import torch
import os
import soundfile as sf  
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import numpy as np


from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from safetensors.torch import load_file
from utils.audio import load_audio

from typing import Any, Dict, Tuple

# sys.path.append('/home/work_nfs23/hyzhang/code/bicodec/bicodec_hw')

sys.path.append('./SparkTTS')

from SparkTTS.sparktts.utils.file import load_config
from SparkTTS.sparktts.modules.speaker.speaker_encoder import SpeakerEncoder
from SparkTTS.sparktts.modules.encoder_decoder.feat_encoder import Encoder
from SparkTTS.sparktts.modules.encoder_decoder.feat_decoder import Decoder
from SparkTTS.sparktts.modules.encoder_decoder.wave_generator import WaveGenerator
from SparkTTS.sparktts.modules.vq.factorized_vector_quantize import FactorizedVectorQuantize

from tqdm import tqdm

class BiCodec(nn.Module):
    def __init__(
        self,
        mel_params: Dict[str, Any],
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: nn.Module,
        speaker_encoder: nn.Module,
        prenet: nn.Module,
        postnet: nn.Module,
        device,
        **kwargs
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.speaker_encoder = speaker_encoder
        self.prenet = prenet
        self.postnet = postnet
        self.init_mel_transformer(mel_params)
        self.device = device
        self.config = load_config('bicodec_ckpt/ckpt/config.yaml')
        self.init_model()
        
    @classmethod
    def load_from_checkpoint(cls, model_dir: Path , device, **kwargs) -> "BiCodec":
        ckpt_path = f'{model_dir}/model.safetensors'
        config = load_config(f'{model_dir}/config.yaml')['audio_tokenizer']
        mel_params = config["mel_params"]
        encoder = Encoder(**config["encoder"])
        quantizer = FactorizedVectorQuantize(**config["quantizer"])
        prenet = Decoder(**config["prenet"])
        postnet = Decoder(**config["postnet"])
        decoder = WaveGenerator(**config["decoder"])
        speaker_encoder = SpeakerEncoder(**config["speaker_encoder"])

        model = cls(
            mel_params=mel_params,
            encoder=encoder,
            decoder=decoder,
            quantizer=quantizer,
            speaker_encoder=speaker_encoder,
            prenet=prenet,
            postnet=postnet,
            device=device
        )

        state_dict = load_file(ckpt_path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        for key in missing_keys:
            print(f"Missing tensor: {key}")
        for key in unexpected_keys:
            print(f"Unexpected tensor: {key}")

        model.eval()
        model.remove_weight_norm()

        return model

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        feat = batch["feat"]
        mel = self.mel_transformer(batch["ref_wav"]).squeeze(1)

        z = self.encoder(feat.transpose(1, 2))
        vq_outputs = self.quantizer(z)

        x_vector, d_vector = self.speaker_encoder(mel.transpose(1, 2))

        conditions = d_vector
        with_speaker_loss = False

        x = self.prenet(vq_outputs["z_q"], conditions)
        pred_feat = self.postnet(x)
        x = x + conditions.unsqueeze(-1)
        wav_recon = self.decoder(x)

        return {
            "vq_loss": vq_outputs["vq_loss"],
            "perplexity": vq_outputs["perplexity"],
            "cluster_size": vq_outputs["active_num"],
            "recons": wav_recon,
            "pred_feat": pred_feat,
            "x_vector": x_vector,
            "d_vector": d_vector,
            "audios": batch["wav"].unsqueeze(1),
            "with_speaker_loss": with_speaker_loss,
        }
        
    def init_model(self):
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.config["wav2vec_model"]
        )
        self.feature_extractor = Wav2Vec2Model.from_pretrained(
            self.config["wav2vec_model"]
        ).to(self.device)
        
        self.feature_extractor.config.output_hidden_states = True
        
    def extract_wav2vec2_features(self, wavs: torch.Tensor) -> torch.Tensor:
        inputs = self.processor(
            wavs,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            output_hidden_states=True,
        ).input_values
        feat = self.feature_extractor(inputs.to(self.feature_extractor.device))
        feats_mix = (
            feat.hidden_states[11] + feat.hidden_states[14] + feat.hidden_states[16]
        ) / 3

        return feats_mix
    def get_ref_clip(self, wav: np.ndarray) -> np.ndarray:
        cfg = self.config["datasets"]
        ref_segment_length = (
            int(cfg["sample_rate"] * cfg["ref_segment_duration"])
            // cfg["latent_hop_length"]
            * cfg["latent_hop_length"]
        )
        wav_length = len(wav)

        if ref_segment_length > wav_length:
            # Repeat and truncate to handle insufficient length
            wav = np.tile(wav, (1 + ref_segment_length) // wav_length)

        return wav[:ref_segment_length]
    
    def process_audio(self, wav_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        wav_16k = load_audio(
            wav_path,
            sampling_rate=16000,
            volume_normalize=True,
        )
        wav_ref = self.get_ref_clip(wav_16k)
        wav_ref = torch.from_numpy(wav_ref).unsqueeze(0).float()

        return wav_ref, wav_16k

    @torch.no_grad()
    def tokenize(self, audio_path):
        ref_wav, wav_16k = self.process_audio(audio_path)
        
        feat = self.extract_wav2vec2_features(wav_16k) 
        mel = self.mel_transformer(ref_wav).squeeze(1).to(self.device)
        self.encoder.to(self.device)
        self.quantizer.to(self.device)
        self.speaker_encoder.to(self.device)
        self.prenet.to(self.device)
        self.decoder.to(self.device)
        
        z = self.encoder(feat.transpose(1, 2))
        semantic_tokens = self.quantizer.tokenize(z)
        global_tokens = self.speaker_encoder.tokenize(mel.transpose(1, 2))

        return semantic_tokens, global_tokens

    @torch.no_grad()
    def detokenize(self, semantic_tokens, global_tokens):      
        z_q = self.quantizer.detokenize(semantic_tokens)

        d_vector = self.speaker_encoder.detokenize(global_tokens)
        x = self.prenet(z_q, d_vector)
        x = x + d_vector.unsqueeze(-1)
        wav_recon = self.decoder(x)

        return wav_recon

    def init_mel_transformer(self, config: Dict[str, Any]):
        import torchaudio.transforms as TT

        self.mel_transformer = TT.MelSpectrogram(
            config["sample_rate"],
            config["n_fft"],
            config["win_length"],
            config["hop_length"],
            config["mel_fmin"],
            config["mel_fmax"],
            n_mels=config["num_mels"],
            power=1,
            norm="slaney",
            mel_scale="slaney",
        )

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                pass  # The module didn't have weight norm

        self.apply(_remove_weight_norm)


def bicodec_model_inference(
    model_path: str,
    meta_lst: str,
    output_dir: str,
    device: str = None,
) -> None:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Audio will be saved to: {output_dir}")

    print("\nStarting to load model and tokenizer...")
    

    bicodec_tokenizer = BiCodec.load_from_checkpoint(
        model_dir="bicodec_ckpt/ckpt",
        device=device
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  
        low_cpu_mem_usage=True,     
        device_map=device           
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        './bicodec_tokenizer'
    )
    model.eval()
    print("Model and tokenizer loaded successfully!")

    speech_generation_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')
    speech_generation_end_id = 8192
    text_generation_start_id = tokenizer.convert_tokens_to_ids('<|TEXT_GENERATION_START|>')
    text_generation_end_id = tokenizer.convert_tokens_to_ids('<|TEXT_GENERATION_END|>')
    text_understanding_start_id = tokenizer.convert_tokens_to_ids('<|TEXT_UNDERSTANDING_START|>')
    text_understanding_end_id = tokenizer.convert_tokens_to_ids('<|TEXT_UNDERSTANDING_END|>')
    speech_understanding_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_START|>')
    speech_understanding_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_END|>')

    speech_global_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GLOBAL_START|>')
    speech_global_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GLOBAL_END|>')
    
    speech_lang_yue = tokenizer.convert_tokens_to_ids('<|SPEECH_LANG_YUE|>')
    speech_lang_norm = tokenizer.convert_tokens_to_ids('<|SPEECH_LANG_NORM|>')


    with open(meta_lst, 'r') as f:
        for line in tqdm(f.readlines()):
            i = line.strip()
            
            parts = i.split('|')
            utt = parts[0]
            tag = parts[1]
            prompt_wav = parts[2]
            text = parts[3]            
                        
            speaker_path = prompt_wav
            
            semantic_tokens, global_token = bicodec_tokenizer.tokenize(speaker_path)
            
            global_token_new = global_token + len(tokenizer) + 8192
            
                
            print(f"\n=== Processing audio ID: {utt} ===")

            text_token = tokenizer(
                text=text,
                add_special_tokens=True, 
                return_tensors=None,  
            )["input_ids"]  # 仅取input_ids（文本token序列）
            
            if(tag == 'yue'):
                tag_token_new = speech_lang_yue
            else:
                tag_token_new = speech_lang_norm
                            
            input_ids = torch.tensor(np.array(  [tag_token_new]
                                              + [speech_global_start_id] 
                                              + np.array(global_token_new.cpu()).flatten().tolist() 
                                              + [speech_global_end_id] 
                                              + [text_understanding_start_id] 
                                              + text_token 
                                              + [text_understanding_end_id] 
                                              + [speech_generation_start_id])).unsqueeze(0)
            input_ids = input_ids.to(device) 
            print("Start generating audio tokens...")
            
            with torch.no_grad():
                generated_tokens = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=2000,  
                    do_sample=True, 
                    temperature=0.8,
                    top_p=0.9,
                    top_k=25,
                    pad_token_id=tokenizer.pad_token_id,  
                    eos_token_id= speech_generation_end_id   
                )
            
            audio_tokens = generated_tokens[:, input_ids.shape[1]:-1] - len(tokenizer)  
            audio_tokens = torch.clamp(audio_tokens, min=0, max=8191)
            
            if audio_tokens.numel() == 0:
                print(f"⚠️ [Skipped] Generation failed (empty tokens), audio ID: {utt}")
                continue

            if audio_tokens.shape[1] == 0:
                print("Generation failed (empty token sequence)")
                continue

            print(f"Generated audio token shape: {audio_tokens.shape}")
            print(f"max {torch.max(audio_tokens)}, min {torch.min(audio_tokens)}")

            if audio_tokens.shape[1] == 0:
                print("Generation failed (empty token sequence)")
                continue

            # -------------------------- 4.4 解码音频Token→Wav --------------------------
            print("Decoding audio tokens and saving WAV file...")
        
            wav = bicodec_tokenizer.detokenize(audio_tokens, global_token)

            wav = wav.detach().cpu().numpy()
            wav = np.squeeze(wav)   
            wav = wav.astype(np.float32)
        
            os.makedirs(f'{output_dir}', exist_ok=True )
            wav_save_path = os.path.join(output_dir, f"{utt}.wav")            
            
            sf.write(
                file=wav_save_path,
                data=wav,
                samplerate=16000  
            )
            print(f"Audio saved successfully: {wav_save_path} (length: {len(wav)/24000:.2f} seconds)")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BiCodec inference script")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the Transformers model (e.g., /path/to/model_dir)"
    )

    parser.add_argument(
        "--meta_path",
        type=str,
        required=True,
        help="Path to the list file containing entries in the format: utt|prompt|prompt_wav|text"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the generated audio files"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Inference device (default: cuda:0)"
    )

    args = parser.parse_args()

    bicodec_model_inference(
        model_path=args.model_path,
        meta_lst=args.meta_path,
        output_dir=args.output_dir,
        device=args.device
    )
