<div align="center">

  # SonicMaster
**SonicMaster: Towards Controllable All-in-One Music Restoration and Mastering**


[![arXiv](https://img.shields.io/badge/arXiv-2508.03448-b31b1b.svg)](http://arxiv.org/abs/2508.03448)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/amaai-lab/)
[![Demo](https://img.shields.io/badge/🎵-Demo-green)](https://huggingface.co/amaai-lab/SonicMaster)
[![Samples Page](https://img.shields.io/badge/Samples-Page-blue)](https://amaai-lab.github.io/SonicMaster/)
[![Dataset](https://img.shields.io/badge/Dataset-download-purple)](https://huggingface.co/datasets/amaai-lab/SonicMasterDataset)


</div>
<div align="center">
<img src="https://ambujmehrish.github.io/SM-Orig/Images/sm.jpeg" alt="SonicMaster" width="400"/>
</div>

## Overview

Music recordings often suffer from audio quality issues such as excessive reverberation, distortion, clipping, tonal imbalances, and a narrowed stereo image, especially when created in non-professional settings without specialized equipment or expertise. These problems are typically corrected using separate specialized tools and manual adjustments. In this paper, we introduce SonicMaster, the first unified generative model for music restoration and mastering that addresses a broad spectrum of audio artifacts with text-based control. SonicMaster is conditioned on natural language instructions to apply targeted enhancements, or can operate in an automatic mode for general restoration.
</div>

<div align="center">
<img src="https://github.com/user-attachments/assets/eb3b799b-04c9-4ff3-bc14-25ce9b74ca16" alt="SonicVerse Architecture" width="800"/>
<p><em>Figure 1: SonicVerse architecture for music captioning with feature detection.</em></p>
</div>

<!--🔥 Live demo available on [Huggingface](https://huggingface.co/spaces/amaai-lab/SonicVerse)-->

## Key Features

- **🎵 Unified Restoration**: All-In-One model to simultaneously handle reverb, clipping, EQ, dynamics, and stereo imbalances.
- **📝 Text-Based Control**: Use natural-language instructions (e.g. “reduce reverb”) for fine-grained audio enhancement.
- **🚀 High-Quality Output**: Objective metrics (FAD, SSIM, etc.) and listening tests show significant quality gains.
- **💾 SonicMaster Dataset**: We release a large-scale dataset of 25k (208 hrs) paired clean and degraded music segments with natural-language prompts for training and evaluation.


## Installation
To run SonicMaster, you should use python==3.13. Then, install the requirements and clone the repo.
```bash
pip install -r requirements_sonic.txt
```

## Training
We trained SonicMaster with pytorch tensor files of our SonicMaster dataset -- for speed. For that, you would first want to pre-encode your audio:
```bash
accelerate launch preencode_latents_acce2.py
```
Then you can start training with the training script that loads pt files from a jsonl metadata file. The script also allows to turn on inference during training (after a certain number of epochs) to monitor your progress.
```bash
accelerate launch train_ptload_inference.py
```

## Citation


If you use SonicMaster in your work, please cite our paper:

_Jan Melechovsky, Ambuj Mehrish, Dorien Herremans. 2025. SonicMaster: Towards Controllable All-in-One Music Restoration and Mastering. ArXiv:2508.03448_

```bibtex
@article{melechovsky2025sonicmaster,
      title={SonicMaster: Towards Controllable All-in-One Music Restoration and Mastering}, 
      author={Jan Melechovsky and Ambuj Mehrish and Dorien Herremans},
      year={2025},
      eprint={2508.03448},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2508.03448}, 
}
```

Read the paper here: [arXiv:2508.0338](http://arxiv.org/abs/2508.03448)

---



<div align="center">
Made with 🎸 by the AMAAI Lab | Singapore
</div>
