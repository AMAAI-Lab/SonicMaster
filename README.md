<div align="center">

  # SonicMaster
**SonicMaster: Towards Controllable All-in-One Music Restoration and Mastering**

[Demo] | [Model]() | [Website and Examples]() | [Paper] | [Dataset](https://huggingface.co/datasets/amaai-lab/SonicMasterDataset)

[![arXiv](https://img.shields.io/badge/arXiv-2508.03448-b31b1b.svg)](http://arxiv.org/abs/2508.03448)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/amaai-lab/SonicMaster)
[![Demo](https://img.shields.io/badge/🎵-Demo-green)](https://huggingface.co/amaai-lab/SonicMaster)
[![Samples Page](https://img.shields.io/badge/Samples-Page-blue)](https://amaai-lab.github.io/SonicMaster/)
[![Dataset](https://img.shields.io/badgeDataset-purple)](https://huggingface.co/datasets/amaai-lab/SonicMasterDataset)


</div>
<div align="center">
<img src="https://ambujmehrish.github.io/SM-Orig/Images/sm.jpeg" alt="SonicMaster" width="400"/>
</div>

## Overview

Music recordings often suffer from audio quality issues such as excessive reverberation, distortion, clipping, tonal imbalances, and a narrowed stereo image, especially when created in non-professional settings without specialized equipment or expertise. These problems are typically corrected using separate specialized tools and manual adjustments. In this paper, we introduce SonicMaster, the first unified generative model for music restoration and mastering that addresses a broad spectrum of audio artifacts with text-based control. SonicMaster is conditioned on natural language instructions to apply targeted enhancements, or can operate in an automatic mode for general restoration.
</div>



## Key Features

- **🎵 Unified Restoration**: All-In-One model to simultaneously handle reverb, clipping, EQ, dynamics, and stereo imbalances.
- **📝 Text-Based Control**: Use natural-language instructions (e.g. “reduce reverb”) for fine-grained audio enhancement.
- **🚀 High-Quality Output**: Objective metrics (FAD, SSIM, etc.) and listening tests show significant quality gains.
- **💾 SonicMaster Dataset**: We release a large-scale dataset of 25k (208 hrs) paired clean and degraded music segments with natural-language prompts for training and evaluation.
