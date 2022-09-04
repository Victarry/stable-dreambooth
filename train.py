import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.pipelines import StableDiffusionPipeline
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms
from datasets import load_dataset

@dataclass
class TrainingConfig:
    image_size = 512 # the generated image resolution
    train_batch_size = 1
    eval_batch_size = 4  # how many images to sample during evaluation
    num_epochs = 250
    gradient_accumulation_steps = 1
    learning_rate = 2e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = 'finetune'  # the model namy locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    guidance_scale = 7.5
    hub_private_repo = False  
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

def pred(model, noisy_latent, time_steps, prompt, guidance_scale):
    batch_size = noisy_latent.shape[0]
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0
    # get unconditional embeddings for classifier free guidance
    if do_classifier_free_guidance:
        max_length = text_input.input_ids.shape[-1]
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latent_model_input = torch.cat([noisy_latent] * 2) if do_classifier_free_guidance else noisy_latent
    time_steps = torch.cat([time_steps] * 2) if do_classifier_free_guidance else time_steps
    noise_pred = model.unet(latent_model_input, time_steps, encoder_hidden_states=text_embeddings)["sample"]
    # perform guidance
    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred
   


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, prompt):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        log_with="tensorboard",
        logging_dir=os.path.join(config.output_dir, "logs")
    )
    if accelerator.is_main_process:
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the 
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader 
    )
    
    global_step = 0


    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch['images']
            # Sample noise to add to the images
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            with torch.no_grad():
                latents = model.vae.encode(clean_images).mode()
            noise = torch.randn(latents.shape).to(clean_images.device)

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = pred(model, noisy_latents, timesteps, prompt, config.guidance_scale)
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                # accelerator.clip_grad_norm_(model.unet.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, model, prompt)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                model.save_pretrained(config.output_dir) 

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def evaluate(config: TrainingConfig, epoch, pipeline: StableDiffusionPipeline, prompt):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    with torch.no_grad():
        images = pipeline(prompt, num_inference_steps=50, width=config.image_size, height=config.image_size, guidance_scale=config.guidance_scale)["sample"]

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


if __name__ == "__main__":
    config = TrainingConfig()
    preprocess = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    config.dataset_name = "imagefolder"
    dataset = load_dataset(config.dataset_name, data_dir="./data", split="train")

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}
    dataset.set_transform(transform)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True)
    
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda"

    model = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(device)
    optimizer = torch.optim.AdamW(model.unet.parameters(), lr=config.learning_rate)
    noise_scheduler = model.scheduler
    prompt = ["a dog"] * config.train_batch_size

    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, prompt=prompt)
