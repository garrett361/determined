import json
import os
import pathlib
from PIL import Image
from typing import Literal, Optional, Sequence, Union


import accelerate
import determined as det
import torch
import torch.nn.functional as F
from determined.pytorch import TorchData
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from torch.utils.data import DataLoader
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

import data


class TextualInversionTrainer:
    """Class for training a textual inversion model."""

    def __init__(
        self,
        use_auth_token: str,
        train_img_dirs: Union[str, Sequence[str]],
        placeholder_tokens: Union[str, Sequence[str]],
        initializer_tokens: Union[str, Sequence[str]],
        learnable_properties: Sequence[Literal["object", "style"]],
        pretrained_model_name_or_path: str = "CompVis/stable-diffusion-v1-4",
        train_batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        optimizer_name: Literal["adam", "adamw", "sgd"] = "sgd",
        learning_rate: float = 5e-04,
        other_optimzer_kwargs: Optional[dict] = None,
        scale_lr: bool = True,
        checkpoint_freq: int = 100,
        metric_report_freq: int = 100,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: Literal["linear", "scaled_linear", "squaredcos_cap_v2"] = "scaled_linear",
        num_train_timesteps: int = 1000,
        train_seed: int = 2147483647,
        img_size: int = 512,
        interpolation: Literal["nearest", "bilinear", "bicubic"] = "bicubic",
        flip_p: float = 0.5,
        center_crop: bool = False,
        inference_prompts: Optional[Union[str, Sequence[str]]] = None,
        inference_noise_scheduler_name: Literal["ddim", "lms-discrete", "pndm"] = "ddim",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator_seed: int = 2147483647,
        other_inference_noise_scheduler_kwargs: Optional[dict] = None,
        latest_checkpoint: Optional[str] = None,
    ) -> None:
        self.use_auth_token = use_auth_token
        self.latest_checkpoint = latest_checkpoint
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        if isinstance(learnable_properties, str):
            learnable_properties = [learnable_properties]
        self.learnable_properties = learnable_properties
        if isinstance(placeholder_tokens, str):
            placeholder_tokens = [placeholder_tokens]
        self.placeholder_tokens = placeholder_tokens
        if isinstance(initializer_tokens, str):
            initializer_tokens = [initializer_tokens]
        self.initializer_tokens = initializer_tokens
        self.img_size = img_size
        self.interpolation = interpolation
        self.flip_p = flip_p
        self.center_crop = center_crop

        self.train_batch_size = train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        assert optimizer_name in (
            "adam",
            "adamw",
            "sgd",
        ), "Optimizer must be one of 'adam', 'adamw' or 'sgd'."
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.other_optimzer_kwargs = other_optimzer_kwargs or {}
        self.scale_lr = scale_lr
        self.checkpoint_freq = checkpoint_freq
        self.metric_report_freq = metric_report_freq
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.num_train_timesteps = num_train_timesteps
        if isinstance(train_img_dirs, str):
            train_img_dirs = [train_img_dirs]
        self.train_img_dirs = train_img_dirs
        self.train_seed = train_seed

        # TODO: Fix inference_prompts default
        self.inference_noise_schedulers = {
            "ddim": DDIMScheduler,
            "lms-discrete": LMSDiscreteScheduler,
            "pndm": PNDMScheduler,
        }
        assert inference_noise_scheduler_name in self.inference_noise_schedulers, (
            f"inference_noise_scheduler must be one {list(self.inference_noise_schedulers.keys())},"
            f" but got {inference_noise_scheduler_name}"
        )

        if isinstance(inference_prompts, str):
            inference_prompts = [inference_prompts]
        self.inference_noise_scheduler_name = inference_noise_scheduler_name
        self.inference_prompts = inference_prompts or [f"a painting of a dog"]
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.generator_seed = generator_seed
        self.other_inference_noise_scheduler_kwargs = other_inference_noise_scheduler_kwargs or {}

        self.logger = accelerate.logging.get_logger(__name__)
        self.steps_completed = 0
        self.loss_history = []
        self.last_mean_loss = None
        self.generated_imgs = {prompt: [] for prompt in self.inference_prompts}

        self.accelerator = accelerate.Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )
        accelerate.utils.set_seed(self.train_seed)

        self.effective_global_batch_size = (
            self.gradient_accumulation_steps
            * self.train_batch_size
            * self.accelerator.num_processes
        )
        # If scale_lr, we linearly scale the bare learning rate by the effective batch size
        if scale_lr:
            self.learning_rate *= self.effective_global_batch_size
            self.logger.info(f"Using scaled learning rate {self.learning_rate}")

        # The below are instantiated through the immediately following methods
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.unet = None
        self.safety_checker = None
        self.feature_extractor = None
        self.placeholder_token_ids = None
        self.train_dataset = None
        self.train_dataloader = None
        self.optimizer = None
        self.train_noise_scheduler = None
        self.original_embedding_idxs = None
        self.original_embedding_tensors = None

        self._build_models()
        self._add_new_tokens()
        self._freeze_layers()
        self._build_dataset_and_dataloader()
        self._build_optimizer()
        self._build_train_noise_scheduler()
        self._wrap_and_prepare()

        # Pipeline construction is deferred until the _save call
        self.inference_noise_scheduler_kwargs = None
        self.pipeline = None

    def train(self) -> None:
        """Run the full latent inversion training loop."""
        self.logger.info("--------------- Starting Training ---------------")
        self.logger.info(f"Effective global batch size: {self.effective_global_batch_size}")
        self.logger.info(f"Learning rate: {self.learning_rate}")
        self.logger.info(f"Train dataset size: {len(self.train_dataset)}")

        try:
            distributed = det.core.DistributedContext.from_torch_distributed()
        except KeyError:
            distributed = None
        with det.core.init(distributed=distributed) as core_context:
            self._restore_latest_checkpoint(core_context)
            # There will be a single op of len max_length, as defined in the searcher config.
            for op in core_context.searcher.operations():
                while self.steps_completed < op.length:
                    for batch in self.train_dataloader:
                        # Use the accumulate method for efficient gradient accumulation.
                        with self.accelerator.accumulate(self.text_encoder):
                            self._train_one_batch(batch)
                        # An SGD step has been taken when self.accelerator.sync_gradients is True.
                        took_sgd_step = self.accelerator.sync_gradients
                        if took_sgd_step:
                            self.steps_completed += 1
                            self.logger.info(f"Step {self.steps_completed} completed")

                            is_end_of_training = self.steps_completed == op.length
                            time_to_report = self.steps_completed % self.metric_report_freq == 0
                            time_to_ckpt = self.steps_completed % self.checkpoint_freq == 0

                            # Report metrics, checkpoint, and preempt as appropriate.
                            if is_end_of_training or time_to_report or time_to_ckpt:
                                self._report_train_metrics(core_context)
                            if is_end_of_training or time_to_ckpt:
                                self._save(core_context)
                                if core_context.preempt.should_preempt():
                                    return

                            if is_end_of_training:
                                break
                if self.accelerator.is_main_process:
                    # Report the final mean loss.
                    op.report_completed(self.last_mean_loss)

    def _train_one_batch(self, batch: TorchData) -> torch.Tensor:
        """Train on a single batch, returning the loss and updating internal metrics."""
        # Convert images to latent space
        latent_dist = self.vae.encode(batch["pixel_values"]).latent_dist
        latents = latent_dist.sample().detach()
        # In 2112.10752, it was found that the latent space variance plays a large role in image
        # quality.  The following scale factor helps to maintain unit latent variance.  See
        # https://github.com/huggingface/diffusers/issues/437 for more details.
        scale_factor = 0.18215
        latents = latents * scale_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn(latents.shape).to(self.accelerator.device)
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.num_train_timesteps,
            (self.train_batch_size,),
            device=self.accelerator.device,
        ).long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.train_noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = F.mse_loss(noise_pred, noise)
        self.accelerator.backward(loss)
        self.loss_history.append(loss.detach())

        # For textual inversion, we only update the embeddings of the newly added concept tokens.
        # This is most safely implemented by copying the original embeddings, rather than zeroing
        # out their gradients, as L2 regularization (for instance) will still modify weights whose
        # gradient is zero. See link below for a discussion:
        # https://discuss.pytorch.org/t/how-to-freeze-a-subset-of-weights-of-a-layer/97498
        last_embedding = (
            self.text_encoder.module.get_input_embeddings().weight.data[-1].detach().clone()
        )
        self.optimizer.step()
        # Only overwrite after the step has actually been taken:
        if self.accelerator.sync_gradients:
            # An extra .module attr is needed due to the accelerator.prepare call.
            token_embeds = self.text_encoder.module.get_input_embeddings().weight.data
            token_embeds[
                self.original_embedding_idxs
            ] = self.original_embedding_tensors.detach().clone()
        self.optimizer.zero_grad()

        return loss

    def _build_models(self) -> None:
        """Download the relevant models using deferred execution:
        https://huggingface.co/docs/accelerate/concept_guides/deferring_execution
        """
        with self.accelerator.main_process_first():
            self.tokenizer = CLIPTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.pretrained_model_name_or_path,
                subfolder="tokenizer",
                use_auth_token=self.use_auth_token,
            )
            self.text_encoder = CLIPTextModel.from_pretrained(
                pretrained_model_name_or_path=self.pretrained_model_name_or_path,
                subfolder="text_encoder",
                use_auth_token=self.use_auth_token,
            )
            self.vae = AutoencoderKL.from_pretrained(
                pretrained_model_name_or_path=self.pretrained_model_name_or_path,
                subfolder="vae",
                use_auth_token=self.use_auth_token,
            )
            self.unet = UNet2DConditionModel.from_pretrained(
                pretrained_model_name_or_path=self.pretrained_model_name_or_path,
                subfolder="unet",
                use_auth_token=self.use_auth_token,
            )
        # Modules for StableDiffusionPipeline only required by chief worker.
        if self.accelerator.is_main_process:
            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                pretrained_model_name_or_path="CompVis/stable-diffusion-safety-checker"
            )
            self.feature_extractor = CLIPFeatureExtractor.from_pretrained(
                pretrained_model_name_or_path="openai/clip-vit-base-patch32"
            )

    def _add_new_tokens(self) -> None:
        """
        Add new concept tokens to the tokenizer.
        """
        # Add the placeholder token in tokenizer
        self.placeholder_token_ids = []
        initializer_token_ids = []
        for placeholder, initializer in zip(self.placeholder_tokens, self.initializer_tokens):
            num_added_tokens = self.tokenizer.add_tokens(placeholder)
            if num_added_tokens == 0:
                raise ValueError(
                    f"Tokenizer already contains the {placeholder}, please choose another token."
                )

            # Convert the initializer_tokens, placeholder_tokens to ids.
            initializer_token_id_list = self.tokenizer.encode(initializer, add_special_tokens=False)
            # Check if initializer_tokens is a single token or a sequence of tokens.
            if len(initializer_token_id_list) > 1:
                raise ValueError(
                    "The initializer token must get mapped to a single id."
                    f" {initializer} is mapped to {initializer_token_id_list}"
                )

            initializer_token_ids.append(initializer_token_id_list[0])
            placeholder_token_id = self.tokenizer.convert_tokens_to_ids(placeholder)
            self.placeholder_token_ids.append(placeholder_token_id)

        # Extend the size of the self.text_encoder to account for the new placeholder_tokens.
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        # Initialize the placeholder vectors to coincide with their initializer vectors.
        token_embeds = self.text_encoder.get_input_embeddings().weight.data
        for p_id, i_id in zip(self.placeholder_token_ids, initializer_token_ids):
            token_embeds[p_id] = token_embeds[i_id]

        # Take a snapshot of the original embedding weights.  Used in the update step to ensure that
        # we only train the newly added concept vectors.
        self.original_embedding_idxs = torch.isin(
            torch.arange(len(self.tokenizer)),
            torch.tensor(self.placeholder_token_ids),
            invert=True,
        )
        self.original_embedding_tensors = (
            token_embeds[self.original_embedding_idxs].detach().clone().to(self.accelerator.device)
        )

    def _freeze_layers(self) -> None:
        """Freeze all not-to-be-trained layers."""
        # Freeze everything and then unfreeze only the layers we want to train.
        for model in (
            self.vae,
            self.unet,
            self.text_encoder,
        ):
            for param in model.parameters():
                param.requires_grad = False

        for param in self.text_encoder.text_model.embeddings.token_embedding.parameters():
            param.requires_grad = True

    def _build_dataset_and_dataloader(self) -> None:
        """Build the dataset and dataloader."""
        self.train_dataset = data.TextualInversionDataset(
            train_img_dirs=self.train_img_dirs,
            tokenizer=self.tokenizer,
            placeholder_tokens=self.placeholder_tokens,
            learnable_properties=self.learnable_properties,
            img_size=self.img_size,
            interpolation=self.interpolation,
            flip_p=self.flip_p,
            center_crop=self.center_crop,
        )
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.train_batch_size, shuffle=True
        )

    def _build_optimizer(self) -> None:
        """Construct the optimizer, recalling that only the embedding vectors are to be trained."""
        embedding_params = self.text_encoder.get_input_embeddings().parameters()
        optim_dict = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}
        self.optimizer = optim_dict[self.optimizer_name](
            embedding_params,  # only optimize the embeddings
            lr=self.learning_rate,
            **self.other_optimzer_kwargs,
        )

    def _build_train_noise_scheduler(self) -> None:
        self.train_noise_scheduler = DDPMScheduler(
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            beta_schedule="scaled_linear",
            num_train_timesteps=self.num_train_timesteps,
            tensor_format="pt",
        )

    def _wrap_and_prepare(self) -> None:
        """Wrap necessary modules for distributed training and set unwrapped, non-trained modules
        to the appropriate eval state."""

        # Freeze the vae and unet completely, and everything in the text encoder except the
        # embedding layer

        self.text_encoder, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.text_encoder, self.optimizer, self.train_dataloader
        )
        self.vae.to(self.accelerator.device)
        self.unet.to(self.accelerator.device)
        self.text_encoder.train()
        self.vae.eval()
        self.unet.eval()

    def _restore_latest_checkpoint(self, core_context: det.core.Context) -> None:
        """Restores the experiment state to the latest saved checkpoint, if it exists."""
        if self.latest_checkpoint is not None:
            with core_context.checkpoint.restore_path(self.latest_checkpoint) as path:
                with self.accelerator.local_main_process_first():
                    with open(path.joinpath("metadata.json"), "r") as f:
                        checkpoint_metadata_dict = json.load(f)
                    self.steps_completed = checkpoint_metadata_dict["steps_completed"]
                    self.generated_imgs = torch.load(path.joinpath("generated_imgs.pt"))
                    optimizer_state_dict = torch.load(path.joinpath("optimizer_state_dict.pt"))
                    self.optimizer.load_state_dict(optimizer_state_dict)
                    learned_embeds_dict = torch.load(path.joinpath("learned_embeds.pt"))
                    # An extra .module attr is needed due to the accelerator.prepare call.
                    token_embeds = self.text_encoder.module.get_input_embeddings().weight.data
                    for idx, tensor in learned_embeds_dict.items():
                        token_embeds[idx] = tensor

    def _save(self, core_context: det.core.Context) -> None:
        """Checkpoints the training state and pipeline."""
        self.logger.info(f"Saving checkpoint at step {self.steps_completed}.")
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            checkpoint_metadata_dict = {
                "steps_completed": self.steps_completed,
                "initializer_tokens": self.initializer_tokens,
                "placeholder_tokens": self.placeholder_tokens,
                "placeholder_token_ids": self.placeholder_token_ids,
                "pretrained_model_name_or_path": self.pretrained_model_name_or_path,
                "inference_noise_scheduler_name": self.inference_noise_scheduler_name,
                "inference_noise_scheduler_kwargs": self.inference_noise_scheduler_kwargs,
            }

            with core_context.checkpoint.store_path(checkpoint_metadata_dict) as (path, storage_id):
                self._build_pipeline()
                self._generate_and_write_imgs(path)
                self._write_optimizer_state_dict_to_path(path)
                self._write_learned_embeddings_to_path(path)

    def _write_optimizer_state_dict_to_path(self, path: pathlib.Path) -> None:
        optimizer_state_dict = self.optimizer.state_dict()
        self.accelerator.save(optimizer_state_dict, path.joinpath("optimizer_state_dict.pt"))

    def _write_learned_embeddings_to_path(self, path: pathlib.Path) -> None:
        learned_embeds = (
            (
                self.accelerator.unwrap_model(self.text_encoder)
                .get_input_embeddings()
                .weight[self.placeholder_token_ids]
            )
            .detach()
            .cpu()
        )
        learned_embeds_dict = {
            idx: tensor for idx, tensor in zip(self.placeholder_token_ids, learned_embeds)
        }
        self.accelerator.save(learned_embeds_dict, path.joinpath("learned_embeds.pt"))

    def _build_pipeline(self) -> None:
        """Build the pipeline for the chief worker only."""
        if self.accelerator.is_main_process:
            inference_noise_scheduler = self.inference_noise_schedulers[
                self.inference_noise_scheduler_name
            ]
            self.inference_noise_scheduler_kwargs = {
                "beta_start": self.beta_start,
                "beta_end": self.beta_end,
                "beta_schedule": self.beta_schedule,
                **self.other_inference_noise_scheduler_kwargs,
            }
            self.pipeline = StableDiffusionPipeline(
                text_encoder=self.accelerator.unwrap_model(self.text_encoder),
                vae=self.vae,
                unet=self.unet,
                tokenizer=self.tokenizer,
                scheduler=inference_noise_scheduler(**self.inference_noise_scheduler_kwargs),
                safety_checker=self.safety_checker,
                feature_extractor=self.feature_extractor,
            ).to(self.accelerator.device)

    def _generate_and_write_imgs(self, path: pathlib.Path) -> None:
        # Generate a new image using the pipeline.
        self.logger.info("Generating sample images")
        imgs_path = path.joinpath("imgs")
        os.makedirs(imgs_path, exist_ok=True)
        for prompt in self.inference_prompts:
            # Fix generator for reproducibility.
            generator = torch.Generator(device=self.accelerator.device).manual_seed(
                self.generator_seed
            )
            generated_img = self.pipeline(
                prompt=prompt,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=generator,
            ).images[0]
            prompt_img_dict = self.generated_imgs[prompt]
            prompt_img_dict.append((self.steps_completed, generated_img))
            prompt_imgs_path = imgs_path.joinpath("_".join(prompt.split()))
            os.makedirs(prompt_imgs_path, exist_ok=True)
            img_grid = Image.new("RGB", size=(len(prompt_img_dict) * self.img_size, self.img_size))
            for idx, (step, img) in enumerate(prompt_img_dict):
                img.save(prompt_imgs_path.joinpath(f"{step}.png"))
                img_grid.paste(img, box=(idx * self.img_size, 0))
            img_grid.save(prompt_imgs_path.joinpath("all_imgs.png"))
            # Create a gif
            first_img = prompt_img_dict[0][1]
            first_img.save(
                fp=prompt_imgs_path.joinpath("all_imgs.gif"),
                format="GIF",
                append_images=(img for _, img in prompt_img_dict),
                save_all=True,
                duration=1000,
                loop=1,
            )
        # Finally, write self.generated_imgs to path for use during a checkpoint restore.
        self.accelerator.save(self.generated_imgs, path.joinpath("generated_imgs.pt"))

    def _report_train_metrics(self, core_context: det.core.Context) -> None:
        """Report training metrics to the Determined master."""
        local_mean_loss = torch.tensor(self.loss_history, device=self.accelerator.device).mean()
        # reduction = 'mean' seems to return the sum rather than the mean:
        self.last_mean_loss = (
            self.accelerator.reduce(local_mean_loss, reduction="sum").item()
            / self.accelerator.num_processes
        )
        self.loss_history = []
        if self.accelerator.is_main_process:
            core_context.train.report_training_metrics(
                steps_completed=self.steps_completed,
                metrics={"loss": self.last_mean_loss},
            )
