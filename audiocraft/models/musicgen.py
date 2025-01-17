# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Main model for using MusicGen. This will combine all the required components
and provide easy access to the generation API.
"""

import os
import typing as tp

import torch

from .encodec import CompressionModel
from .lm import LMModel
from .builders import get_debug_compression_model, get_debug_lm_model
from .loaders import load_compression_model, load_lm_model, HF_MODEL_CHECKPOINTS_MAP
from ..data.audio_utils import convert_audio
from ..modules.conditioners import ConditioningAttributes, WavCondition
from ..utils.autocast import TorchAutocast


MelodyList = tp.List[tp.Optional[torch.Tensor]]
MelodyType = tp.Union[torch.Tensor, MelodyList]

def wrap_around_slice(tensor, start, end):
    if end <= tensor.shape[2]:
        return tensor[:, :, start:end]
    else:
        part1 = tensor[:, :, start:]
        part2 = tensor[:, :, :(end % tensor.shape[2])]
        return torch.cat([part1, part2], dim=2)



class MusicGen:
    """MusicGen main model with convenient generation API.

    Args:
        name (str): name of the model.
        compression_model (CompressionModel): Compression model
            used to map audio to invertible discrete representations.
        lm (LMModel): Language model over discrete representations.
    """
    def __init__(self, name: str, compression_model: CompressionModel, lm: LMModel):
        self.name = name
        self.compression_model = compression_model
        self.lm = lm
        self.device = next(iter(lm.parameters())).device
        self.generation_params: dict = {}
        self.set_generation_params(duration=15)  # 15 seconds by default
        if self.device.type == 'cpu':
            self.autocast = TorchAutocast(enabled=False)
        else:
            self.autocast = TorchAutocast(
                enabled=True, device_type=self.device.type, dtype=torch.float16)

    @property
    def frame_rate(self) -> int:
        """Roughly the number of AR steps per seconds."""
        return self.compression_model.frame_rate

    @property
    def sample_rate(self) -> int:
        """Sample rate of the generated audio."""
        return self.compression_model.sample_rate

    @property
    def audio_channels(self) -> int:
        """Audio channels of the generated audio."""
        return self.compression_model.channels

    @staticmethod
    def get_pretrained(name: str = 'melody', device='cuda'):
        """Return pretrained model, we provide four models:
        - small (300M), text to music, # see: https://huggingface.co/facebook/musicgen-small
        - medium (1.5B), text to music, # see: https://huggingface.co/facebook/musicgen-medium
        - melody (1.5B) text to music and text+melody to music, # see: https://huggingface.co/facebook/musicgen-melody
        - large (3.3B), text to music, # see: https://huggingface.co/facebook/musicgen-large
        """

        if name == 'debug':
            # used only for unit tests
            compression_model = get_debug_compression_model(device)
            lm = get_debug_lm_model(device)
            return MusicGen(name, compression_model, lm)

        if name not in HF_MODEL_CHECKPOINTS_MAP:
            raise ValueError(
                f"{name} is not a valid checkpoint name. "
                f"Choose one of {', '.join(HF_MODEL_CHECKPOINTS_MAP.keys())}"
            )

        cache_dir = os.environ.get('MUSICGEN_ROOT', None)
        compression_model = load_compression_model(name, device=device, cache_dir=cache_dir)
        lm = load_lm_model(name, device=device, cache_dir=cache_dir)
        if name == 'melody':
            lm.condition_provider.conditioners['self_wav'].match_len_on_eval = True

        return MusicGen(name, compression_model, lm)

    def set_generation_params(self, use_sampling: bool = True, top_k: int = 250,
                              top_p: float = 0.0, temperature: float = 1.0,
                              duration: float = 30.0, cfg_coef: float = 3.0,
                              two_step_cfg: bool = False, wav_and_text_separate: bool = False,
                              wav_cfg_proportion: float = 0.5):
        """Set the generation parameters for MusicGen.

        Args:
            use_sampling (bool, optional): Use sampling if True, else do argmax decoding. Defaults to True.
            top_k (int, optional): top_k used for sampling. Defaults to 250.
            top_p (float, optional): top_p used for sampling, when set to 0 top_k is used. Defaults to 0.0.
            temperature (float, optional): Softmax temperature parameter. Defaults to 1.0.
            duration (float, optional): Duration of the generated waveform. Defaults to 30.0.
            cfg_coef (float, optional): Coefficient used for classifier free guidance. Defaults to 3.0.
            two_step_cfg (bool, optional): If True, performs 2 forward for Classifier Free Guidance,
                instead of batching together the two. This has some impact on how things
                are padded but seems to have little impact in practice.
        """
        self.generation_params = {
            'max_gen_len': int(duration * self.frame_rate),
            'use_sampling': use_sampling,
            'temp': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'cfg_coef': cfg_coef,
            'two_step_cfg': two_step_cfg,
            'wav_and_text_separate': wav_and_text_separate,
            'wav_cfg_proportion': wav_cfg_proportion
        }

    def generate_unconditional(self, num_samples: int, progress: bool = False) -> torch.Tensor:
        """Generate samples in an unconditional manner.

        Args:
            num_samples (int): Number of samples to be generated.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        descriptions: tp.List[tp.Optional[str]] = [None] * num_samples
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, None)
        return self._generate_tokens(attributes, prompt_tokens, progress)

    def generate(self, descriptions: tp.List[str], progress: bool = False) -> torch.Tensor:
        """Generate samples conditioned on text.

        Args:
            descriptions (tp.List[str]): A list of strings used as text conditioning.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, None)
        assert prompt_tokens is None
        return self._generate_tokens(attributes, prompt_tokens, progress)
    def generate_continuation(self, prompt: torch.Tensor, prompt_sample_rate: int,
                              descriptions: tp.Optional[tp.List[tp.Optional[str]]] = None,
                              progress: bool = False) -> torch.Tensor:
        """Generate samples conditioned on audio prompts.

        Args:
            prompt (torch.Tensor): A batch of waveforms used for continuation.
                Prompt should be [B, C, T], or [C, T] if only one sample is generated.
            prompt_sample_rate (int): Sampling rate of the given audio waveforms.
            descriptions (tp.List[str], optional): A list of strings used as text conditioning. Defaults to None.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        if prompt.dim() == 2:
            prompt = prompt[None]
        if prompt.dim() != 3:
            raise ValueError("prompt should have 3 dimensions: [B, C, T] (C = 1).")
        prompt = convert_audio(prompt, prompt_sample_rate, self.sample_rate, self.audio_channels)
        if descriptions is None:
            descriptions = [None] * len(prompt)
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, prompt)
        assert prompt_tokens is not None
        return self._generate_tokens(attributes, prompt_tokens, progress)

    def generate_with_chroma(self, descriptions: tp.List[str], melody_wavs: MelodyType,
                             melody_sample_rate: int, progress: bool = False) -> torch.Tensor:
        """Generate samples conditioned on text and melody.

        Args:
            descriptions (tp.List[str]): A list of strings used as text conditioning.
            melody_wavs: (torch.Tensor or list of Tensor): A batch of waveforms used as
                melody conditioning. Should have shape [B, C, T] with B matching the description length,
                C=1 or 2. It can be [C, T] if there is a single description. It can also be
                a list of [C, T] tensors.
            melody_sample_rate: (int): Sample rate of the melody waveforms.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        if isinstance(melody_wavs, torch.Tensor):
            if melody_wavs.dim() == 2:
                melody_wavs = melody_wavs[None]
            if melody_wavs.dim() != 3:
                raise ValueError("Melody wavs should have a shape [B, C, T].")
            melody_wavs = list(melody_wavs)
        else:
            for melody in melody_wavs:
                if melody is not None:
                    assert melody.dim() == 2, "One melody in the list has the wrong number of dims."

        melody_wavs = [
            convert_audio(wav, melody_sample_rate, self.sample_rate, self.audio_channels)
            if wav is not None else None
            for wav in melody_wavs]
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions=descriptions, prompt=None,
                                                                        melody_wavs=melody_wavs)
        assert prompt_tokens is None
        return self._generate_tokens(attributes, prompt_tokens, progress)

    def generate_continuation_with_melody(self, prompt: torch.Tensor, prompt_sample_rate: int, melody_wavs: MelodyType, melody_sample_rate: int,
                              descriptions: tp.Optional[tp.List[tp.Optional[str]]] = None,
                              progress: bool = False) -> torch.Tensor:
        """Generate samples conditioned on audio prompts and melody.

        Args:
            prompt (torch.Tensor): A batch of waveforms used for continuation.
                Prompt should be [B, C, T], or [C, T] if only one sample is generated.
            prompt_sample_rate (int): Sampling rate of the given audio waveforms.
            melody_wavs: (torch.Tensor or list of Tensor): A batch of waveforms used as
                melody conditioning. Should have shape [B, C, T] with B matching the description length,
                C=1 or 2. It can be [C, T] if there is a single description. It can also be
                a list of [C, T] tensors.
            melody_sample_rate: (int): Sample rate of the melody waveforms.
            descriptions (tp.List[str], optional): A list of strings used as text conditioning. Defaults to None.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        if prompt.dim() == 2:
            prompt = prompt[None]
        if prompt.dim() != 3:
            raise ValueError("prompt should have 3 dimensions: [B, C, T] (C = 1).")
        prompt = convert_audio(prompt, prompt_sample_rate, self.sample_rate, self.audio_channels)

        if isinstance(melody_wavs, torch.Tensor):
            if melody_wavs.dim() == 2:
                melody_wavs = melody_wavs[None]
            if melody_wavs.dim() != 3:
                raise ValueError("Melody wavs should have a shape [B, C, T].")
            melody_wavs = list(melody_wavs)
        else:
            for melody in melody_wavs:
                if melody is not None:
                    assert melody.dim() == 2, "One melody in the list has the wrong number of dims."
        melody_wavs = [
            convert_audio(wav, melody_sample_rate, self.sample_rate, self.audio_channels)
            if wav is not None else None
            for wav in melody_wavs]

        if descriptions is None:
            descriptions = [None] * len(prompt)

        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions=descriptions, prompt=prompt, melody_wavs=melody_wavs)
        assert prompt_tokens is not None
        return self._generate_tokens(attributes, prompt_tokens, progress)

    
    def generate_music_for_duration(self, description: str, melody: tp.Optional[MelodyType], melody_sr: tp.Optional[float], window_len_secs: float, total_duration_secs: float, slide_secs: float, progress: bool = True) -> torch.Tensor:
        """
        Generate music for a longer duration using the MusicGen model.

        Args:
            description (str): The description to condition the model.
            melody (tp.Optional[MelodyType]): The melody to follow if there is one.
            melody_sr (tp.Optional[Float]): The sample rate of the input melody.
            window_len_secs (float): How long each generation should be individually, in seconds.
            total_duration_secs (float): The total duration for which music should be generated, in seconds.
            slide_secs (float): The duration by which the window should slide after each generation, in seconds.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to True.

        Returns:
            torch.Tensor: Generated audio, of shape [B, C, T], T is defined by the generation params.
        """
        if window_len_secs > 30:
            raise ValueError("MusicGen is absolutely not capable of generating past 30 seconds. Don't do it. Seriously.")
        self.set_generation_params(use_sampling=self.generation_params['use_sampling'], top_k=self.generation_params['top_k'],
                              top_p=self.generation_params['top_p'], temperature=self.generation_params['temp'],
                              duration=window_len_secs, cfg_coef=self.generation_params['cfg_coef'],
                              two_step_cfg=self.generation_params['two_step_cfg'], wav_and_text_separate=self.generation_params['wav_and_text_separate'],
                              wav_cfg_proportion=self.generation_params['wav_cfg_proportion'])
        sample_rate = self.sample_rate  # get the sample rate
        slide_duration_frames = slide_secs * sample_rate  # slide duration in frames

        # Create a list to store the generated sections
        sections = []
        # Generate the first section
        if melody is None:
            section = self.generate([description], progress=progress)
        else:
            section = self.generate_with_chroma([description], melody_wavs=melody, melody_sample_rate=melody_sr, progress=progress)
        sections.append(section)
        print("Section shape after first generate: ", section.shape)  # This will tell you the shape of the initial section


        # Generate subsequent sections
        while (len(sections) - 1) * (window_len_secs-slide_secs) + window_len_secs < total_duration_secs:
            # Concatenate all sections
            full_music = torch.cat(sections, axis=-1)

            # Get the last slide_seconds from the full_music as the prompt for the next section
            prompt = full_music[:, :, -slide_secs*sample_rate:]
            print("Prompt shape: ", prompt.shape)

            # Generate next section with or without melody
            if melody is None:
                section = self.generate_continuation(prompt, sample_rate, descriptions=[description], progress=progress)
            else:
                # Calculate the start and end points for the melody slice
                current_time_in_secs = (len(sections) - 1) * (window_len_secs-slide_secs) + window_len_secs
                start_frame = int(current_time_in_secs * melody_sr)
                end_frame = start_frame + int(window_len_secs * melody_sr)
                # Slice the melody tensor according to the current time position and wrap around if its too long for the melody tensor
                melody_slice = wrap_around_slice(melody, start_frame, end_frame)

                section = self.generate_continuation_with_melody(prompt, sample_rate, melody_wavs=melody_slice, melody_sample_rate=melody_sr, descriptions=[description], progress=progress)
            print("Section shape before slicing: ", section.shape)
            section = section[:, :, int(slide_secs*sample_rate):]
            print("Section shape after slicing: ", section.shape)
            
            sections.append(section)

        # Concatenate all sections one final time to make sure all of them are included
        full_music = torch.cat(sections, axis=-1)
        print("full_music shape after concatenation: ", full_music.shape)  # This will tell you the shape of the full music tensor after concatenation

        return full_music


    @torch.no_grad()
    def _prepare_tokens_and_attributes(
            self,
            descriptions: tp.Sequence[tp.Optional[str]],
            prompt: tp.Optional[torch.Tensor],
            melody_wavs: tp.Optional[MelodyList] = None,
    ) -> tp.Tuple[tp.List[ConditioningAttributes], tp.Optional[torch.Tensor]]:
        """Prepare model inputs.

        Args:
            descriptions (tp.List[str]): A list of strings used as text conditioning.
            prompt (torch.Tensor): A batch of waveforms used for continuation.
            melody_wavs (tp.Optional[torch.Tensor], optional): A batch of waveforms
                used as melody conditioning. Defaults to None.
        """
        attributes = [
            ConditioningAttributes(text={'description': description})
            for description in descriptions]

        if melody_wavs is None:
            for attr in attributes:
                attr.wav['self_wav'] = WavCondition(
                    torch.zeros((1, 1), device=self.device),
                    torch.tensor([0], device=self.device),
                    path='null_wav')  # type: ignore
        else:
            if self.name != "melody":
                raise RuntimeError("This model doesn't support melody conditioning. "
                                   "Use the `melody` model.")
            assert len(melody_wavs) == len(descriptions), \
                f"number of melody wavs must match number of descriptions! " \
                f"got melody len={len(melody_wavs)}, and descriptions len={len(descriptions)}"
            for attr, melody in zip(attributes, melody_wavs):
                if melody is None:
                    attr.wav['self_wav'] = WavCondition(
                        torch.zeros((1, 1), device=self.device),
                        torch.tensor([0], device=self.device),
                        path='null_wav')  # type: ignore
                else:
                    attr.wav['self_wav'] = WavCondition(
                        melody.to(device=self.device),
                        torch.tensor([melody.shape[-1]], device=self.device))

        if prompt is not None:
            if descriptions is not None:
                assert len(descriptions) == len(prompt), "Prompt and nb. descriptions doesn't match"
            prompt = prompt.to(self.device)
            prompt_tokens, scale = self.compression_model.encode(prompt)
            assert scale is None
        else:
            prompt_tokens = None
        return attributes, prompt_tokens

    def _generate_tokens(self, attributes: tp.List[ConditioningAttributes],
                         prompt_tokens: tp.Optional[torch.Tensor], progress: bool = False) -> torch.Tensor:
        """Generate discrete audio tokens given audio prompt and/or conditions.

        Args:
            attributes (tp.List[ConditioningAttributes]): Conditions used for generation (text/melody).
            prompt_tokens (tp.Optional[torch.Tensor]): Audio prompt used for continuation.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        Returns:
            torch.Tensor: Generated audio, of shape [B, C, T], T is defined by the generation params.
        """
        def _progress_callback(generated_tokens: int, tokens_to_generate: int):
            print(f'{generated_tokens: 6d} / {tokens_to_generate: 6d}', end='\r')

        if prompt_tokens is not None:
            assert self.generation_params['max_gen_len'] >= prompt_tokens.shape[-1], \
                "Prompt is longer than audio to generate"

        callback = None
        if progress:
            callback = _progress_callback

        # generate by sampling from LM
        with self.autocast:
            gen_tokens = self.lm.generate(prompt_tokens, attributes, callback=callback, **self.generation_params)

        # generate audio
        assert gen_tokens.dim() == 3
        with torch.no_grad():
            gen_audio = self.compression_model.decode(gen_tokens, None)
        return gen_audio
