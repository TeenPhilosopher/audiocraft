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
                              two_step_cfg: bool = False):
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

    
    def generate_music_for_duration(self, description: str, melody: Optional[Tuple[int, np.ndarray]], window_len_secs: float, total_duration_seconds: float, slide_seconds: float, progress: bool = True) -> torch.Tensor:
        """
        Generate music for a longer duration using the MusicGen model.

        Args:
            description (str): The description to condition the model.
            melody (Optional[Tuple[int, np.ndarray]]): A tuple of sample rate and the melody waveform. None if no melody is provided.
            window_len_secs (float): How long each generation should be individually, in seconds.
            total_duration_seconds (float): The total duration for which music should be generated, in seconds.
            slide_seconds (float): The duration by which the window should slide after each generation, in seconds.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to True.

        Returns:
            torch.Tensor: Generated audio, of shape [B, C, T], T is defined by the generation params.
        """
        if window_len_secs > 30:
            raise ValueError("MusicGen is absolutely not capable of generating past 30 seconds. Don't do it. Seriously.")
        self.set_generation_params(duration=window_len_secs)
        sample_rate = self.sample_rate  # get the sample rate
        slide_duration_frames = slide_seconds * sample_rate  # slide duration in frames

        # Create a list to store the generated sections
        sections = []

        # Convert melody to the right format if it is provided
        if melody:
            melody_sr, melody_data = melody
            melody_tensor = torch.from_numpy(melody_data).to(self.device).float().t().unsqueeze(0)
            if melody_tensor.dim() == 2:
                melody_tensor = melody_tensor[None]
            melody_tensor = melody_tensor[..., :int(melody_sr * self.lm.cfg.dataset.segment_duration)]
        else:
            melody_tensor = None

        # Generate the first section
        if melody_tensor is None:
            section = self.generate([description], progress=progress)
        else:
            section = self.generate_with_chroma([description], melody_wavs=melody_tensor, melody_sample_rate=melody_sr, progress=progress)
        sections.append(section)

        # Generate subsequent sections
        while len(sections) * slide_seconds < total_duration_seconds:
            # Get the last window_len_secs from the previous section as the prompt for the next section
            prompt = sections[-1][:, :, -window_len_secs*sample_rate:]

            # Generate next section with or without melody
            if melody_tensor is None:
                section = self.generate_continuation(prompt, sample_rate, descriptions=[description], progress=progress)
            else:
                # Calculate the start and end points for the melody slice
                start_frame = int(len(sections) * slide_seconds * sample_rate)
                end_frame = start_frame + int(window_len_secs * sample_rate)
                # Slice the melody tensor according to the current time position
                melody_slice = melody_tensor[:, :, start_frame:end_frame]
                section = self.generate_continuation_with_melody(prompt, sample_rate, melody_wavs=melody_slice, melody_sample_rate=melody_sr, descriptions=[description], progress=progress)
            section = section[:, :, int(slide_seconds*sample_rate):]
            sections.append(section)


        # Concatenate all sections
        full_music = torch.cat(sections, axis=-1)

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
            assert self.generation_params['max_gen_len'] > prompt_tokens.shape[-1], \
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
