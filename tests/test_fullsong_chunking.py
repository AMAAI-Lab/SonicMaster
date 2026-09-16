import ast
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fullsong_chunking import (
    CARRY_LATENT_FRAMES,
    MAIN_LATENT_FRAMES,
    SONICMASTER_SAMPLE_RATE,
    TRAINED_DURATION_SECONDS,
    FullSongGeometry,
    crossfade_and_trim,
    make_overlapping_chunks,
    make_vae_native_geometry,
)


class FullSongGeometryTests(unittest.TestCase):
    def test_published_checkpoint_contract(self):
        self.assertEqual(SONICMASTER_SAMPLE_RATE, 44_100)
        self.assertEqual(MAIN_LATENT_FRAMES, 645)
        self.assertEqual(CARRY_LATENT_FRAMES, MAIN_LATENT_FRAMES // 3)
        self.assertEqual(CARRY_LATENT_FRAMES, 215)
        self.assertEqual(TRAINED_DURATION_SECONDS, 30)

    def test_vae_native_waveform_geometry(self):
        geometry = make_vae_native_geometry(2048, 645, 44_100)
        self.assertEqual(geometry.chunk_size, 1_320_960)
        self.assertEqual(geometry.overlap, 440_320)
        self.assertEqual(geometry.stride, 880_640)
        self.assertEqual(geometry.chunk_size % geometry.hop_length, 0)
        self.assertEqual(geometry.overlap % geometry.hop_length, 0)

    def test_invalid_checkpoint_contract_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "hop length"):
            make_vae_native_geometry(0, 645, 44_100)
        with self.assertRaisesRegex(RuntimeError, "sequence length"):
            make_vae_native_geometry(2048, 646, 44_100)
        with self.assertRaisesRegex(RuntimeError, "44100 Hz VAE"):
            make_vae_native_geometry(2048, 645, 48_000)


class FullSongStitchingTests(unittest.TestCase):
    geometry = FullSongGeometry(
        hop_length=1,
        chunk_size=30_000,
        overlap=10_000,
        stride=20_000,
    )

    def test_empty_audio_produces_no_chunks(self):
        audio = torch.empty(2, 0)
        self.assertEqual(make_overlapping_chunks(audio, self.geometry), [])

    def test_short_audio_is_padded_then_trimmed_exactly(self):
        source = torch.randn(2, 1_000)
        chunks = make_overlapping_chunks(source, self.geometry)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].shape, (2, self.geometry.chunk_size))
        result = crossfade_and_trim(
            [chunks[0].unsqueeze(0)], self.geometry, source.shape[1]
        )
        self.assertEqual(result.shape, (1, 2, 1_000))
        torch.testing.assert_close(result.squeeze(0), source)

    def test_exact_stride_boundary_is_preserved(self):
        source = torch.randn(2, self.geometry.stride)
        chunks = make_overlapping_chunks(source, self.geometry)
        self.assertEqual(len(chunks), 1)
        result = crossfade_and_trim(
            [chunks[0].unsqueeze(0)], self.geometry, source.shape[1]
        )
        torch.testing.assert_close(result.squeeze(0), source)

    def test_multi_chunk_track_preserves_stereo_and_boundaries(self):
        samples = 65_000
        t = torch.arange(samples, dtype=torch.float32)
        source = torch.stack((torch.sin(t * 0.013), torch.cos(t * 0.017)))
        chunks = make_overlapping_chunks(source, self.geometry)
        decoded = [chunk.unsqueeze(0) for chunk in chunks]
        result = crossfade_and_trim(decoded, self.geometry, samples)
        self.assertEqual(result.shape, (1, 2, samples))
        torch.testing.assert_close(result.squeeze(0), source, rtol=1e-6, atol=1e-6)
        self.assertTrue(torch.equal(result[0, :, 0], source[:, 0]))
        self.assertTrue(torch.equal(result[0, :, -1], source[:, -1]))

    def test_final_trim_cannot_hide_stitched_underflow(self):
        decoded = [torch.zeros(1, 2, 999)]
        with self.assertRaisesRegex(RuntimeError, "shorter than the source"):
            crossfade_and_trim(decoded, self.geometry, 1_000)

    def test_crossfade_rejects_chunk_shorter_than_overlap(self):
        decoded = [
            torch.zeros(1, 2, self.geometry.chunk_size),
            torch.zeros(1, 2, self.geometry.overlap - 1),
        ]
        with self.assertRaisesRegex(RuntimeError, "shorter than the VAE-native overlap"):
            crossfade_and_trim(decoded, self.geometry, self.geometry.chunk_size)


class FullSongEntryPointTests(unittest.TestCase):
    required_imports = {
        "CARRY_LATENT_FRAMES",
        "MAIN_LATENT_FRAMES",
        "SONICMASTER_SAMPLE_RATE",
        "TRAINED_DURATION_SECONDS",
        "crossfade_and_trim",
        "make_overlapping_chunks",
        "make_vae_native_geometry",
    }

    def entry_point_source(self, filename):
        return (ROOT / filename).read_text(encoding="utf-8")

    def imported_helper_names(self, source):
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "fullsong_chunking":
                return {alias.name for alias in node.names}
        return set()

    def assert_shared_contract(self, filename):
        source = self.entry_point_source(filename)
        self.assertTrue(self.required_imports <= self.imported_helper_names(source))
        self.assertIn("make_vae_native_geometry(", source)
        self.assertIn("make_overlapping_chunks(", source)
        self.assertIn("crossfade_and_trim(", source)
        self.assertIn("duration=TRAINED_DURATION_SECONDS", source)
        self.assertIn("VAE main encode length mismatch", source)
        self.assertIn("VAE decoded chunk length mismatch", source)
        self.assertIn("VAE carry encode length mismatch", source)

    def test_infer_single_uses_shared_contract_and_has_no_false_cli_knobs(self):
        source = self.entry_point_source("infer_single.py")
        self.assert_shared_contract("infer_single.py")
        self.assertNotIn('"--fs"', source)
        self.assertNotIn('"--chunk_duration"', source)
        self.assertNotIn('"--overlap_duration"', source)
        self.assertIn("torchaudio.functional.resample", source)
        self.assertIn("SONICMASTER_SAMPLE_RATE", source)

    def test_dataset_fullsong_path_uses_shared_contract(self):
        source = self.entry_point_source("inference_fullsong.py")
        self.assert_shared_contract("inference_fullsong.py")
        self.assertIn("fs = SONICMASTER_SAMPLE_RATE", source)
        self.assertIn("Expected {fs} Hz", source)


if __name__ == "__main__":
    unittest.main()
