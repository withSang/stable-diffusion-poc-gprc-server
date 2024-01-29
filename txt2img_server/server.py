import io
import os

import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from imwatermark import WatermarkEncoder

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
# grpc server related codes
from concurrent import futures
import grpc
from protos import generate_image_pb2, generate_image_pb2_grpc
from config import ModelConfig, ServerConfig

torch.set_grad_enabled(False)

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if device == torch.device("cuda"):
        model.cuda()
    elif device == torch.device("cpu"):
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")
    model.eval()
    return model

def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def run_inference(model, sampler, wm_encoder, prompt="a robot waving arms"):
    batch_size = ModelConfig.n_samples
    n_rows = ModelConfig.n_rows if ModelConfig.n_rows > 0 else batch_size

    start_code = None

    precision_scope = autocast if ModelConfig.precision=="autocast" else nullcontext

    assert prompt is not None
    data = [batch_size * [prompt]]

    with torch.no_grad(), \
        precision_scope(ModelConfig.device), \
        model.ema_scope():
            all_samples = list()
            for n in trange(ModelConfig.n_iter, desc="Sampling"):
                for prompts in data:
                    uc = None
                    if ModelConfig.scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)
                    shape = [ModelConfig.C, ModelConfig.H // ModelConfig.f, ModelConfig.W // ModelConfig.f]
                    samples, _ = sampler.sample(S=ModelConfig.steps,
                                                     conditioning=c,
                                                     batch_size=ModelConfig.n_samples,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=ModelConfig.scale,
                                                     unconditional_conditioning=uc,
                                                     eta=ModelConfig.ddim_eta,
                                                     x_T=start_code)

                    x_samples = model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    for x_sample in x_samples:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        img = put_watermark(img, wm_encoder)
                        # img.save(f"test-{n}.png")

                    all_samples.append(x_samples)

            # additionally, save as grid
            grid = torch.stack(all_samples, 0)
            grid = rearrange(grid, 'n b c h w -> (n b) c h w')
            grid = make_grid(grid, nrow=n_rows)

            # to image
            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            grid = Image.fromarray(grid.astype(np.uint8))
            grid = put_watermark(grid, wm_encoder)
            # grid.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
            return grid


class ImageGenerationServicer(generate_image_pb2_grpc.ImageGenerationServicer):
    def __init__(self, model, sampler, wm_encoder):
        self.model = model
        self.sampler = sampler
        self.wm_encoder = wm_encoder
        super().__init__()

    def GenerateImage(self, request, context):
        image = run_inference(self.model, self.sampler, self.wm_encoder, prompt=request.prompt)
        width, height = image.size

        byteIO = io.BytesIO()
        image.save(byteIO, format='PNG')
        image_bytes = byteIO.getvalue()

        return generate_image_pb2.GenerateImageResponse(
            success=True,
            image=generate_image_pb2.Image(
                width=width,
                height=height,
                data=image_bytes,
            )
        )

def serve():
    # Load model.
    print("Loading model...")
    seed_everything(ModelConfig.seed)
    config = OmegaConf.load(f"{ModelConfig.config}")
    device = torch.device("cuda") if ModelConfig.device == "cuda" else torch.device("cpu")
    model = load_model_from_config(config, f"{ModelConfig.ckpt}", device)

    sampler = DDIMSampler(model, device=device)

    # print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    # wm = "SDV2"
    # wm_encoder = WatermarkEncoder()
    # wm_encoder.set_watermark('bytes', wm.encode('utf-8'))
    wm_encoder = None

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    generate_image_pb2_grpc.add_ImageGenerationServicer_to_server(
        ImageGenerationServicer(model=model, sampler=sampler, wm_encoder=wm_encoder), server
    )
    server.add_insecure_port(f'[::]:{ServerConfig.PORT}')
    server.start()
    print(f'Server started on port {ServerConfig.PORT}')
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
