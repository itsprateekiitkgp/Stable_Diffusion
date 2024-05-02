import torch
from diffusers import StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
import numpy as np

model_id = "runwayml/stable-diffusion-v1-5"
generator = torch.manual_seed(12345)


def txt2img(prompt, save_path):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
    pipe = pipe.to("cuda")
    image = pipe(prompt, num_inference_steps=50, generator=generator).images[0]
    image.save(save_path)


def txt_depth2img(depth_img, prompt, save_path):
    controlnet_depth = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(model_id,
                                                             controlnet=controlnet_depth, torch_dtype=torch.float16,
                                                             safety_checker=None)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    image = pipe(prompt, num_inference_steps=20, generator=generator, image=depth_img).images[0]
    image.save(save_path)


def txt_normal2img(normal_img, prompt, save_path):
    # controlnet_normal = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-normal", torch_dtype=torch.float16)
    controlnet_normal = ControlNetModel.from_pretrained(
        "fusing/stable-diffusion-v1-5-controlnet-normal", torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(model_id,
                                                             controlnet=controlnet_normal,
                                                             torch_dtype=torch.float16,
                                                             safety_checker=None)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    image = pipe(prompt, num_inference_steps=20, generator=generator, image=normal_img).images[0]
    image.save(save_path)


def txt_depth_normal2img(depth_img, normal_img, prompt, save_path):
    controlnet_depth = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)
    controlnet_normal = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-normal", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(model_id,
                                                             controlnet=[controlnet_normal, controlnet_depth],
                                                             torch_dtype=torch.float16,
                                                             safety_checker=None)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    image = pipe(prompt, num_inference_steps=50, generator=generator, image=[normal_img, depth_img]).images[0]
    image.save(save_path)


