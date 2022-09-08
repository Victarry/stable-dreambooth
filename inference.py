from pathlib import Path
from diffusers.pipelines import StableDiffusionPipeline
import torch
from argparse import ArgumentParser
import json

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--save_dir", default="outputs")
    parser.add_argument("--sample_nums", default=16)
    parser.add_argument("-gs", "--guidance_scale", type=float, default=7.5)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    with open(Path(args.checkpoint_dir) / 'config.json') as f:
        config = json.loads(f.read())
    args.prompt = args.prompt.replace("[V]", config["identifier"])
    device = "cuda"
    model = StableDiffusionPipeline.from_pretrained(args.checkpoint_dir).to(device)

    with torch.no_grad():
        with torch.autocast("cuda"):
            images = model([args.prompt] * args.sample_nums, height=512, width=512, guidance_scale=args.guidance_scale, num_inference_steps=50)["sample"]
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for i, image in enumerate(images):
        image.save(save_dir / f'{i}.jpg')
