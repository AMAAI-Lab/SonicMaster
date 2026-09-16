"""Shared VAE-native geometry and stitching for full-song inference."""

from dataclasses import dataclass

import torch


SONICMASTER_SAMPLE_RATE = 44_100
MAIN_LATENT_FRAMES = 645
CARRY_LATENT_FRAMES = MAIN_LATENT_FRAMES // 3
TRAINED_DURATION_SECONDS = 30


@dataclass(frozen=True)
class FullSongGeometry:
    """Waveform geometry corresponding to the published checkpoint latents."""

    hop_length: int
    chunk_size: int
    overlap: int
    stride: int


def make_vae_native_geometry(
    vae_hop_length, model_audio_seq_len, vae_sample_rate
):
    """Validate the published checkpoint contract and return sample geometry."""
    hop_length = int(vae_hop_length)
    if hop_length <= 0:
        raise RuntimeError(f"Invalid VAE hop length: {hop_length}")
    if int(model_audio_seq_len) != MAIN_LATENT_FRAMES:
        raise RuntimeError(
            "VAE-native chunking requires the trained SonicMaster sequence "
            f"length {MAIN_LATENT_FRAMES}, got {model_audio_seq_len}."
        )
    if int(vae_sample_rate) != SONICMASTER_SAMPLE_RATE:
        raise RuntimeError(
            "The published SonicMaster checkpoint requires a "
            f"{SONICMASTER_SAMPLE_RATE} Hz VAE, got {vae_sample_rate}."
        )

    chunk_size = MAIN_LATENT_FRAMES * hop_length
    overlap = CARRY_LATENT_FRAMES * hop_length
    if overlap <= 0 or overlap >= chunk_size:
        raise RuntimeError(
            "VAE-native overlap must be positive and smaller than the chunk."
        )
    return FullSongGeometry(
        hop_length=hop_length,
        chunk_size=chunk_size,
        overlap=overlap,
        stride=chunk_size - overlap,
    )


def make_overlapping_chunks(audio, geometry):
    """Return right-padded ``[channels, chunk_size]`` waveform chunks."""
    chunks = []
    start = 0
    total = audio.shape[1]
    while start < total:
        end = min(start + geometry.chunk_size, total)
        chunk = audio[:, start:end]
        if chunk.shape[1] < geometry.chunk_size:
            chunk = torch.nn.functional.pad(
                chunk, (0, geometry.chunk_size - chunk.shape[1])
            )
        chunks.append(chunk)
        start += geometry.stride
    return chunks


def crossfade_and_trim(decoded_chunks, geometry, target_length):
    """Linear-crossfade chunks and trim padding without hiding underflow."""
    if not decoded_chunks:
        raise ValueError("decoded_chunks must not be empty")
    if target_length < 0:
        raise ValueError("target_length must not be negative")

    final = decoded_chunks[0]
    for current in decoded_chunks[1:]:
        if final.shape[-1] < geometry.overlap:
            raise RuntimeError(
                "Previous stitched waveform is shorter than the VAE-native overlap."
            )
        if current.shape[-1] < geometry.overlap:
            raise RuntimeError(
                "Decoded chunk is shorter than the VAE-native overlap."
            )
        previous_overlap = final[:, :, -geometry.overlap :]
        current_overlap = current[:, :, : geometry.overlap]
        alpha = torch.linspace(
            1.0,
            0.0,
            steps=geometry.overlap,
            dtype=previous_overlap.dtype,
            device=previous_overlap.device,
        ).view(1, 1, -1)
        blended = previous_overlap * alpha + current_overlap * (1.0 - alpha)
        final = torch.cat(
            [
                final[:, :, : -geometry.overlap],
                blended,
                current[:, :, geometry.overlap :],
            ],
            dim=2,
        )

    if target_length > final.shape[-1]:
        raise RuntimeError(
            "Stitched waveform is shorter than the source; refusing to hide a "
            f"length error with final trim ({final.shape[-1]} < {target_length})."
        )
    return final[:, :, :target_length]
