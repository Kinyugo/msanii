from argparse import ArgumentParser
from typing import Optional

import gradio as gr
import matplotlib
import numpy as np
import torch
from omegaconf import OmegaConf

from ..config import DemoConfig
from ..pipeline import Pipeline
from .helpers import (
    run_audio2audio,
    run_inpainting,
    run_interpolation,
    run_outpainting,
    run_sampling,
)

HEADER = """
<div style="display:flex; gap: 1em;">
    <a target="_blank" href="https://arxiv.org/abs/2301.06468"><img
        src="https://img.shields.io/badge/arXiv-2301.06468-<COLOR>.svg" alt="arXiv">
    </a>
    <a target="_blank" href="https://github.com/Kinyugo/msanii">
        <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/Kinyugo/msanii?style=social">
    </a>
    <a target="_blank" href="https://huggingface.co/spaces/kinyugo/msanii" rel="noopener noreferrer"><img
        src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue" alt="Hugging Face Spaces">
    </a>
    <a target="_blank" href="https://colab.research.google.com/github/Kinyugo/msanii/blob/main/notebooks/msanii_demo.ipynb">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a>
</div>

# Msanii: High Fidelity Music Synthesis on a Shoestring Budget
"""


def run_demo(config: DemoConfig) -> Optional[gr.Blocks]:
    # -------------------------------------------
    # Configure Matplotlib
    # -------------------------------------------
    # Prevents pixelated fonts on figures
    matplotlib.use("webagg")
    matplotlib.style.use(["seaborn", "fast"])

    # -------------------------------------------
    # Load Pipeline checkpoint
    # -------------------------------------------
    pipeline = Pipeline.from_pretrained(config.ckpt_path)
    pipeline = pipeline.to(torch.device(config.device))
    pipeline = pipeline.to(getattr(torch, config.dtype))

    # -------------------------------------------
    # Define gradio interface
    # -------------------------------------------
    with gr.Blocks(title="Msanii") as demo:
        gr.Markdown(HEADER)

        with gr.Row():
            # Main Section
            with gr.Column(scale=2):
                with gr.Tab("Sampling"):
                    sampling_spectrogram_output = gr.Plot(label="Spectrogram")
                    sampling_waveform_output = gr.Plot(label="Waveform")
                    sampling_audio_output = gr.Audio(label="Sample Audio")
                    sampling_button = gr.Button(value="Run", variant="primary")

                with gr.Tab("Audio2Audio"):
                    a2a_audio_input = gr.Audio(label="Source Audio")
                    a2a_spectrogram_output = gr.Plot(label="Spectrogram")
                    a2a_waveform_output = gr.Plot(label="Waveform")
                    a2a_audio_output = gr.Audio(label="Sample Audio")
                    a2a_button = gr.Button(value="Run", variant="primary")

                with gr.Tab("Interpolation"):
                    interpolation_first_audio_input = gr.Audio(
                        label="First Source Audio"
                    )
                    interpolation_second_audio_input = gr.Audio(
                        label="Second Source Audio"
                    )
                    interpolation_spectrogram_output = gr.Plot(label="Spectrogram")
                    interpolation_waveform_output = gr.Plot(label="Waveform")
                    interpolation_audio_output = gr.Audio(label="Sample Audio")
                    interpolation_button = gr.Button(value="Run", variant="primary")

                with gr.Tab("Inpainting"):
                    inpainting_audio_input = gr.Audio(label="Source Audio")
                    inpainting_mask_input = gr.Text(
                        label="Mask Intervals (seconds) e.g: 20-30,50-60"
                    )
                    inpainting_spectrogram_output = gr.Plot(label="Spectrogram")
                    inpainting_waveform_output = gr.Plot(label="Waveform")
                    inpainting_audio_output = gr.Audio(label="Sample Audio")
                    inpainting_button = gr.Button(value="Run", variant="primary")
                with gr.Tab("Outpainting"):
                    outpainting_audio_input = gr.Audio(label="Source Audio")
                    outpainting_spectrogram_output = gr.Plot(label="Spectrogram")
                    outpainting_waveform_output = gr.Plot(label="Waveform")
                    outpainting_audio_output = gr.Audio(label="Sample Audio")
                    outpainting_button = gr.Button(value="Run", variant="primary")

            # Options
            with gr.Column(scale=1):
                with gr.Accordion(label="General Options", open=True):
                    with gr.Group():
                        inference_steps_slider = gr.Slider(
                            minimum=1,
                            maximum=pipeline.scheduler.num_train_timesteps,
                            value=20,
                            step=1,
                            label="Number of Inference Steps",
                        )
                        griffin_lim_iters_slider = gr.Slider(
                            minimum=1,
                            maximum=1000,
                            value=100,
                            step=1,
                            label="Number of GriffinLim Iterations",
                        )
                        seed = gr.Number(
                            value=lambda: np.random.randint(0, 1_000_000), label="Seed"
                        )

                with gr.Accordion(label="Task Specific Options", open=False):
                    with gr.Group():
                        gr.Markdown("Sampling Options")
                        duration_slider = gr.Slider(
                            minimum=10, maximum=190, label="Audio Duration (seconds)"
                        )
                        channels_slider = gr.Slider(
                            minimum=1,
                            maximum=2,
                            step=1,
                            value=2,
                            label="Audio Channels",
                        )

                    with gr.Group():
                        gr.Markdown("Audio2Audio & Interpolation Options")
                        ratio_slider = gr.Slider(
                            minimum=0, maximum=1, value=0.5, label="Interpolation Ratio"
                        )
                        strength_slider = gr.Slider(
                            minimum=0, maximum=1, value=0.1, label="Noise Strength"
                        )

                    with gr.Group():
                        gr.Markdown("Inpainting & Outpainting")
                        jump_length_slider = gr.Slider(
                            minimum=1,
                            maximum=pipeline.scheduler.num_train_timesteps,
                            value=10,
                            label="Number of Forward Steps",
                        )
                        jump_n_samples_slider = gr.Slider(
                            minimum=1,
                            maximum=pipeline.scheduler.num_train_timesteps,
                            value=10,
                            label="Number of Forward Jumps",
                        )

                    with gr.Group():
                        gr.Markdown("Outpainting")
                        num_spans_slider = gr.Slider(
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=5,
                            label="Number of Outpaint Spans (1/2 duration)",
                        )
                with gr.Accordion(label="Advanced Options", open=False):
                    use_nv_checkbox = gr.Checkbox(
                        value=True, label="Use Neural Vocoder"
                    )
                    eta_slider = gr.Slider(minimum=0, maximum=1, label="eta")
                    max_abs_value_slider = gr.Slider(
                        minimum=0, maximum=1, value=0.05, label="Maximum Absolute Value"
                    )
                    verbose_checkbox = gr.Checkbox(value=False, label="Verbose")

        sampling_button.click(
            lambda *args: run_sampling(pipeline, *args),
            inputs=[
                duration_slider,
                channels_slider,
                inference_steps_slider,
                eta_slider,
                use_nv_checkbox,
                griffin_lim_iters_slider,
                seed,
                verbose_checkbox,
            ],
            outputs=[
                sampling_spectrogram_output,
                sampling_waveform_output,
                sampling_audio_output,
            ],
        )

        a2a_button.click(
            lambda *args: run_audio2audio(pipeline, *args),
            inputs=[
                a2a_audio_input,
                inference_steps_slider,
                strength_slider,
                use_nv_checkbox,
                griffin_lim_iters_slider,
                seed,
                eta_slider,
                max_abs_value_slider,
                verbose_checkbox,
            ],
            outputs=[a2a_spectrogram_output, a2a_waveform_output, a2a_audio_output],
        )

        interpolation_button.click(
            lambda *args: run_interpolation(pipeline, *args),
            inputs=[
                interpolation_first_audio_input,
                interpolation_second_audio_input,
                inference_steps_slider,
                ratio_slider,
                strength_slider,
                use_nv_checkbox,
                griffin_lim_iters_slider,
                seed,
                eta_slider,
                max_abs_value_slider,
                verbose_checkbox,
            ],
            outputs=[
                interpolation_spectrogram_output,
                interpolation_waveform_output,
                interpolation_audio_output,
            ],
        )

        inpainting_button.click(
            lambda *args: run_inpainting(pipeline, *args),
            inputs=[
                inpainting_audio_input,
                inpainting_mask_input,
                jump_length_slider,
                jump_n_samples_slider,
                inference_steps_slider,
                use_nv_checkbox,
                griffin_lim_iters_slider,
                seed,
                eta_slider,
                max_abs_value_slider,
                verbose_checkbox,
            ],
            outputs=[
                inpainting_spectrogram_output,
                inpainting_waveform_output,
                inpainting_audio_output,
            ],
        )

        outpainting_button.click(
            lambda *args: run_outpainting(pipeline, *args),
            inputs=[
                outpainting_audio_input,
                num_spans_slider,
                inference_steps_slider,
                use_nv_checkbox,
                griffin_lim_iters_slider,
                seed,
                eta_slider,
                max_abs_value_slider,
                verbose_checkbox,
            ],
            outputs=[
                outpainting_spectrogram_output,
                outpainting_waveform_output,
                outpainting_audio_output,
            ],
        )

    if config.launch:
        demo.launch(debug=True)
    else:
        return demo


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config_path", help="path to config file", type=str)
    args = parser.parse_args()

    default_demo_config = OmegaConf.structured(DemoConfig)
    file_demo_config = OmegaConf.load(args.config_path)
    demo_config = OmegaConf.merge(default_demo_config, file_demo_config)

    run_demo(demo_config)
