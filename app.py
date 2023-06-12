"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from tempfile import NamedTemporaryFile
import argparse
import torch
import gradio as gr
import os
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

MODEL = None
IS_SHARED_SPACE = "musicgen/MusicGen" in os.environ.get('SPACE_ID', '')


def load_model(version):
    print("Loading model", version)
    return MusicGen.get_pretrained(version)


def predict(model, text, melody, window_len_secs, total_duration_secs, slide_secs, topk, topp, temperature, cfg_coef,wav_and_text_separate,wav_cfg_proportion):
    global MODEL
    topk = int(topk)
    if MODEL is None or MODEL.name != model:
        MODEL = load_model(model)
    print(MODEL.lm.fuser)
    print(MODEL.lm.fuser.fuse2cond)
    MODEL.set_generation_params(use_sampling=True, top_k=topk,
                              top_p=topp, temperature=temperature,
                              duration=window_len_secs, cfg_coef=cfg_coef,
                              wav_and_text_separate=wav_and_text_separate,
                              wav_cfg_proportion=wav_cfg_proportion)

    if melody:
        sr, melody = melody[0], torch.from_numpy(melody[1]).to(MODEL.device).float().t().unsqueeze(0)
        print(melody.shape)
        if melody.dim() == 2:
            melody = melody[None]
        melody = melody[..., :int(sr * total_duration_secs)]
    else:
        sr, melody = None, None
    
    output = MODEL.generate_music_for_duration(description=text, melody=melody, melody_sr=sr, window_len_secs=window_len_secs, total_duration_secs=total_duration_secs, slide_secs=slide_secs)
    output = output.detach().cpu().float()[0]
    with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
        audio_write(
            file.name, output, MODEL.sample_rate, strategy="loudness",
            loudness_headroom_db=16, loudness_compressor=True, add_suffix=False)
        waveform_video = gr.make_waveform(file.name)
    return waveform_video


def ui(**kwargs):
    with gr.Blocks() as interface:
        gr.Markdown(
            """
            # MusicGen
            
The current application serves as an enhanced version of MusicGen, incorporating the long-form generation capabilities highlighted in the research paper, but not provided in the original code by Facebook. It is essential to understand that the generation of long-form content using this approach will inevitably be slower on a per-second basis compared to producing clips shorter than 30 seconds.

The reason for this is that the algorithm ingests a certain amount of previously generated audio (defaulted to 20 seconds) and then generates continuations. As a result, following the first thirty seconds, the model generates audio clips with a length equal to 30 seconds minus the sliding_secs parameter (sliding_secs is defaulted to 20 seconds, so this means that we default to generating 10-second continuations). This mechanism ensures temporal consistency, although it decreases the efficiency of long-term generation by a factor of three, asymptotically. However, you may adjust the sliding_secs parameter for faster generation, with a potential compromise on temporal consistency.

It is crucial to note that modifying the source code to enable direct generation of clips exceeding 30-second length is not advisable. Although the source code allows for straightforward removal of this limit, the audio quality significantly deteriorates past the 30-second point, often resulting in silence or unpleasant screeching white noise after 35 or 40 seconds. This is not an artificial constraint imposed by Meta; it is a limitation of the model's training, which does not cater to longer clips.

Nevertheless, despite its slower pace, this approach provides a viable solution to the issue, yielding high-quality audio output.
            """
        )
        if IS_SHARED_SPACE:
            gr.Markdown("""
                Pew pew pew
                """)
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    text = gr.Text(label="Input Text", interactive=True)
                    melody = gr.Audio(source="upload", type="numpy", label="Melody Condition (optional)", interactive=True)
                with gr.Row():
                    submit = gr.Button("Submit")
                with gr.Row():
                    model = gr.Radio(["melody", "medium", "small", "large"], label="Model", value="melody", interactive=True)
                with gr.Row():
                    window_len_secs = gr.Slider(minimum=1, maximum=30, value=30, label="Window length (secs)", interactive=True)
                    total_duration_secs = gr.Slider(minimum=1, maximum=1000, value=70, label="Total duration (secs)", interactive=True)
                    slide_secs = gr.Slider(minimum=1, maximum=30, value=20, label="Slide (secs)", interactive=True)

                with gr.Row():
                    topk = gr.Number(label="Top-k", value=250, interactive=True)
                    topp = gr.Number(label="Top-p", value=0, interactive=True)
                    temperature = gr.Number(label="Temperature", value=1.0, interactive=True)
                    cfg_coef = gr.Number(label="Classifier Free Guidance", value=3.0, interactive=True)
                with gr.Row():
                    wav_and_text_separate = gr.Checkbox(label="Separate Wav and Text Conditions", value=False, interactive=True)
                    wav_cfg_proportion = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, label="Wav CFG Proportion (valid if 'Separate Wav and Text Conditions' is checked)", interactive=True)

            with gr.Column():
                output = gr.Video(label="Generated Music")
        submit.click(predict, inputs=[model, text, melody, window_len_secs, total_duration_secs, slide_secs, topk, topp, temperature, cfg_coef, wav_and_text_separate,wav_cfg_proportion], outputs=[output])

        gr.Examples(
            fn=predict,
            examples=[
                [
                    "An 80s driving pop song with heavy drums and synth pads in the background",
                    "./assets/bach.mp3",
                    "melody"
                ],
                [
                    "A cheerful country song with acoustic guitars",
                    "./assets/bolero_ravel.mp3",
                    "melody"
                ],
                [
                    "90s rock song with electric guitar and heavy drums",
                    None,
                    "medium"
                ],
                [
                    "a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions",
                    "./assets/bach.mp3",
                    "melody"
                ],
                [
                    "lofi slow bpm electro chill with organic samples",
                    None,
                    "medium",
                ],
            ],
            inputs=[text, melody, model],
            outputs=[output]
        )
        gr.Markdown(
            """
            ### More details

            The model will generate a short (or long!) music extract based on the description you provided.
            You can generate up to INFINITE seconds of audio because screw 30 second limits.

            We present 4 model variations:
            1. Melody -- a music generation model capable of generating music condition on text and melody inputs. **Note**, you can also use text only.
            2. Small -- a 300M transformer decoder conditioned on text only.
            3. Medium -- a 1.5B transformer decoder conditioned on text only.
            4. Large -- a 3.3B transformer decoder conditioned on text only (might OOM for the longest sequences.)

            When using `melody`, ou can optionaly provide a reference audio from
            which a broad melody will be extracted. The model will then try to follow both the description and melody provided.

            You can also use your own GPU or a Google Colab by following the instructions on our repo.
            See [github.com/facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)
            for more details.
            """
        )

        # Show the interface
        launch_kwargs = {}
        username = kwargs.get('username')
        password = kwargs.get('password')
        server_port = kwargs.get('server_port', 0)
        inbrowser = kwargs.get('inbrowser', False)
        share = kwargs.get('share', False)
        server_name = kwargs.get('listen')

        launch_kwargs['server_name'] = server_name

        if username and password:
            launch_kwargs['auth'] = (username, password)
        if server_port > 0:
            launch_kwargs['server_port'] = server_port
        if inbrowser:
            launch_kwargs['inbrowser'] = inbrowser
        if share:
            launch_kwargs['share'] = share

        interface.queue().launch(**launch_kwargs, max_threads=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--listen',
        type=str,
        default='127.0.0.1',
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=0,
        help='Port to run the server listener on',
    )
    parser.add_argument(
        '--inbrowser', action='store_true', help='Open in browser'
    )
    parser.add_argument(
        '--share', action='store_true', help='Share the gradio UI'
    )

    args = parser.parse_args()

    ui(
        username=args.username,
        password=args.password,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        share=args.share,
        listen=args.listen
    )
