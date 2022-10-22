import random
from pathlib import Path

import torch

from diffusers import StableDiffusionPipeline

prompt = "parrot with mohawk"
num_samples = 10
num_inference_steps = 50
use_finetuned = True
img_size = { "height": 128, "width": 128 }

output_dir = Path("output")
prompt_dir = output_dir / prompt
prompt_dir.mkdir(exist_ok=True)

if use_finetuned:
    kwargs = {"pretrained_model_name_or_path": "experiments/sd-slackmojis-model-fname"}
else:
    kwargs = {"pretrained_model_name_or_path": "CompVis/stable-diffusion-v1-4", "revision": "fp16", "torch_dtype": torch.float16}

pipe = StableDiffusionPipeline.from_pretrained(**kwargs)
pipe.to("cuda")
pipe.safety_checker = lambda images, clip_input: (images, False)

for i in range(num_samples):
    generator = torch.Generator("cuda").manual_seed(random.randint(0, 1024))

    image = pipe(prompt=prompt, **img_size, num_inference_steps=num_inference_steps, generator=generator).images[0]
    image.save(prompt_dir / f"{prompt}_{i}.png")