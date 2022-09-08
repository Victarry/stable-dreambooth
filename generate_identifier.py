from diffusers.pipelines import StableDiffusionPipeline
import torch
import random 

if __name__ == "__main__":
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda"
    model = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True, cache_dir="./.cache").to(device)
    tokenizer = model.tokenizer

    rare_tokens = []
    for k, v in tokenizer.encoder.items():
        if len(k) <= 3 and 40000 > v > 35000:
            rare_tokens.append(k)
    
    
    identifiers = []
    for _ in range(3):
        idx = random.randint(0, len(rare_tokens))
        identifiers.append(rare_tokens[idx])
    
    print(" ".join(identifiers))