import logging
import os
from dataclasses import dataclass, field
from typing import List

import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    StableDiffusion3Pipeline,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TextToImageInput(BaseModel):
    model: str
    prompt: str
    size: str | None = None
    n: int | None = None


@dataclass
class PresetModels:
    SD3: List[str] = field(
        default_factory=lambda: ["stabilityai/stable-diffusion-3-medium"]
    )
    SD3_5: List[str] = field(
        default_factory=lambda: [
            "stabilityai/stable-diffusion-3.5-large",
            "stabilityai/stable-diffusion-3.5-large-turbo",
            "stabilityai/stable-diffusion-3.5-medium",
        ]
    )


class TextToImagePipelineSD3:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.pipeline: StableDiffusion3Pipeline | None = None
        self.device: str | None = None

    def start(self):
        if torch.cuda.is_available():
            model_path = self.model_path or "stabilityai/stable-diffusion-3.5-large"
            logger.info("Loading CUDA")
            self.device = "cuda"
            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
            ).to(device=self.device)
        elif torch.backends.mps.is_available():
            model_path = self.model_path or "stabilityai/stable-diffusion-3.5-medium"
            logger.info("Loading MPS for Mac M Series")
            self.device = "mps"
            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            ).to(device=self.device)
        else:
            raise Exception("No CUDA or MPS device available")


class ModelPipelineInitializer:
    def __init__(self, model: str = "", type_models: str = "t2im"):
        self.model = model
        self.type_models = type_models
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "mps"
        self.model_type = None

    def initialize_pipeline(self):
        if not self.model:
            raise ValueError("Model name not provided")

        # Create appropriate pipeline based on model type and type_models
        self.pipeline = TextToImagePipelineSD3(self.model)

        return self.pipeline
