"""Image-only character-consistency test (needs ComfyUI; skipped otherwise).

Generates ONE character reference (Krea T2I, fixed seed), then reuses it
for 3 scene images through the character-reference workflow. No video,
no TTS, no FFmpeg. Run:  python -m unittest discover -s tests -k TestCharacterLive
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path

from providers.comfy import ComfyClient
from providers.image.base import ImageRequest
from providers.registry import create_provider

from tests import COMFY_URL, require_comfy

AI_VIDEO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(AI_VIDEO_DIR, "projects", "character_test")

CHARACTER = (
    "a young Vietnamese woman with long black hair wearing a white ao dai"
)
REF_PROMPT = (
    f"Photorealistic portrait of {CHARACTER}, gentle smile, soft studio light, "
    f"plain warm-gray background, head and shoulders, highly detailed"
)
SCENES = (
    ("scene_01_hoian.png",
     f"The same {CHARACTER}, walking through the lantern-lit streets of "
     f"Hoi An at sunset, glowing lanterns, cinematic photography"),
    ("scene_02_river.png",
     f"The same {CHARACTER}, standing beside the Thu Bon River at dusk, "
     f"wooden boats, reflections on the water, cinematic photography"),
    ("scene_03_cafe.png",
     f"The same {CHARACTER}, sitting at a traditional Vietnamese cafe at "
     f"night, warm lamps, street life behind, cinematic photography"),
)


class TestCharacterLive(unittest.TestCase):
    def test_character_consistency_images(self):
        require_comfy(self)
        os.makedirs(OUT_DIR, exist_ok=True)
        provider = create_provider("image", AI_VIDEO_DIR, "krea",
                                   client=ComfyClient(COMFY_URL))
        ref_path = os.path.join(OUT_DIR, "character_reference.png")
        ref = provider.generate(ImageRequest(
            prompt=REF_PROMPT, negative_prompt="", width=736, height=1312,
            seed=11, output_path=Path(ref_path)))
        self.assertTrue(os.path.isfile(ref_path))
        print(f"\nREFERENCE: {ref_path} ({ref.width}x{ref.height})")
        made = []
        for i, (fname, prompt) in enumerate(SCENES, start=101):
            dest = os.path.join(OUT_DIR, fname)
            result = provider.generate(ImageRequest(
                prompt=prompt, negative_prompt="", width=736, height=1312,
                seed=i, output_path=Path(dest), reference_image_path=Path(ref_path)))
            self.assertTrue(os.path.isfile(dest))
            made.append((dest, result.width, result.height))
            print(f"SCENE: {dest} ({result.width}x{result.height}) seed={i}")
        self.assertEqual(len(made), 3)
        # scene outputs must inherit the reference dimensions (img2img
        # preserves the VAE-encoded reference size)
        from providers.image.krea import KreaImageProvider
        ref_dims = KreaImageProvider.probe_image_size(ref_path)
        for dest, w, h in made:
            self.assertEqual(
                KreaImageProvider.probe_image_size(dest), ref_dims,
                f"{dest} dims differ from reference")


if __name__ == "__main__":
    unittest.main()
