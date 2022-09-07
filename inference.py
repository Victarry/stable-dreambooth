from pathlib import Path
from diffusers.pipelines import StableDiffusionPipeline
import torch
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--save_dir", default="outputs")
    parser.add_argument("--sample_nums", default=4)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda"
    model = StableDiffusionPipeline.from_pretrained(args.checkpoint_dir).to(device)

    with torch.no_grad():
        images = model([args.prompt] * args.sample_nums, height=512, width=512, guidance_scale=7.5, num_inference_steps=50)["sample"]
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for i, image in enumerate(images):
        image.save(save_dir / f'{i}.jpg')