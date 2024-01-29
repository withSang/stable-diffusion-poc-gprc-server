import os
from dotenv import load_dotenv

basedir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
load_dotenv(os.path.join(basedir, '.env'))  # load environment variables from .env file

class ModelConfig:
    """configuration about model."""
    seed = 42
    steps = 50
    ddim_eta = 0.0
    n_iter = 3
    H = 512
    W = 512
    C = 4
    f = 8
    n_samples = 3
    n_rows = 0
    scale = 9.0
    config="configs/stable-diffusion/v2-inference-v.yaml"
    ckpt = os.environ.get("MODEL_PATH") or "768model.ckpt"
    precision="autocast"
    repeat=1
    device = "cuda" # or "cpu"

class ServerConfig:
    """configuration about gRPC."""
    PORT = int(os.environ.get("PORT") or "50051")
