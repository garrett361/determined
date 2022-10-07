import json
import logging
import os
import pathlib
from contextlib import nullcontext
from PIL import Image
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union


import determined as det
import torch
import torch.nn as nn
from determined.experimental import client
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import pil_to_tensor
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from detsd import utils

NOISE_SCHEDULER_DICT = {
    "ddim": DDIMScheduler,
    "lms-discrete": LMSDiscreteScheduler,
    "pndm": PNDMScheduler,
}
DEFAULT_SCHEDULER_KWARGS_DICT = {
    "pndm": {"skip_prk_steps": True},
    "ddim": {"clip_sample": False},
    "lms-discrete": {},
}


class DetSDTextualInversionPipeline:
    """Class for generating images from a Stable Diffusion checkpoint trained using Determined
    AI. Initialize with no arguments in order to run plan Stable Diffusion without any trained
    textual inversion embeddings.
    """

    def __init__(
        self,
        learned_embeddings_filename: str = "learned_embeddings_dict.pt",
        scheduler_name: Literal["ddim", "lms-discrete", "pndm"] = "pndm",
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: Literal["linear", "scaled_linear", "squaredcos_cap_v2"] = "scaled_linear",
        other_scheduler_kwargs: Optional[Dict[str, Any]] = None,
        pretrained_model_name_or_path: str = "CompVis/stable-diffusion-v1-4",
        device: str = "cuda",
        use_autocast: bool = False,
        use_fp16: bool = False,
    ) -> None:
        # We assume that the Huggingface User Access token has been stored as the HF_AUTH_TOKEN
        # environment variable. See https://huggingface.co/docs/hub/security-tokens
        try:
            self.use_auth_token = os.environ["HF_AUTH_TOKEN"]
        except KeyError:
            raise KeyError(
                "Please set your HF User Access token as the HF_AUTH_TOKEN environment variable."
            )
        self.learned_embeddings_filename = learned_embeddings_filename
        self.scheduler_name = scheduler_name
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.other_scheduler_kwargs = (
            other_scheduler_kwargs or DEFAULT_SCHEDULER_KWARGS_DICT[scheduler_name]
        )
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.device = device
        if use_fp16 and not use_autocast:
            raise ValueError("If use_fp16 is True, use_autocast must also be True.")
        self.use_autocast = use_autocast
        self.use_fp16 = use_fp16

        scheduler_kwargs = {
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "beta_schedule": self.beta_schedule,
            **self.other_scheduler_kwargs,
        }
        self.scheduler = NOISE_SCHEDULER_DICT[self.scheduler_name](**scheduler_kwargs)

        # The below attrs are non-trivially instantiated as necessary through appropriate methods.
        self.all_checkpoint_paths = []
        self.learned_embeddings_dict = {}
        self.concept_to_dummy_tokens_map = {}
        self.all_added_concepts = []

        self._build_models()
        self._build_pipeline()

    @classmethod
    def generate_on_cluster(cls) -> None:
        # TODO Clean up generally, report number of generated images at training data, save & restore.
        """Creates a DetSDTextualInversionPipeline instance on the cluster, drawing hyperparameters
        and other needed information from the Determined master, and then generates images. Expects
        the `hyperparameters` section of the config to be broken into the following sections:
        - `pipeline`: containing all __init__ args
        - `uuids`: a (possibly empty) array of any checkpoint UUIDs which are to be loaded into
        the pipeline.
        - `call`: all arguments which are to be passed to the `__call__` method which generates
        images.
        - `save_freq`: an integer specifying how often to write images to tensorboard.
        """
        info = det.get_cluster_info()
        assert info is not None, "generate_on_cluster() must be called on a Determined cluster."
        hparams = info.trial.hparams
        latest_checkpoint = info.latest_checkpoint

        # Extract relevant groups from hparams.
        pipeline_init_kwargs = hparams["pipeline"]
        uuid_list = hparams["uuids"]
        call_kwargs = hparams["call"]
        save_freq = hparams["save_freq"]

        logger = logging.getLogger(__name__)

        # Get the distributed context, as needed.
        try:
            distributed = det.core.DistributedContext.from_torch_distributed()
        except KeyError:
            distributed = None

        with det.core.init(
            distributed=distributed, tensorboard_mode=det.core.TensorboardMode.MANUAL
        ) as core_context:
            logger.info("--------------- Generating Images ---------------")

            # Get worker data.
            process_index = core_context.distributed.get_rank()
            is_main_process = process_index == 0
            num_processes = core_context.distributed.get_size()

            # Choose random seeds to be unique, to avoid repeated images.
            assert (
                "seed" in call_kwargs
            ), "`seed` must be specified when calling `generate_on_cluster`."
            call_kwargs["seed"] += process_index

            # TODO: support CPU case?
            device = f"cuda:{process_index}" if distributed is not None else "cuda"
            pipeline_init_kwargs["device"] = device

            # Instantiate the pipeline and load in any checkpoints by uuid.
            pipeline = cls(**pipeline_init_kwargs)
            pipeline.load_from_uuids(uuid_list)

            # Tensorboard writer.
            tb_dir = core_context.train.get_tensorboard_path()
            tb_writer = SummaryWriter(log_dir=tb_dir)
            # Use the __call__ args apart for the tensorboard tag.
            tb_tag = ", ".join([f"{k}: {v}" for k, v in call_kwargs.items() if v])

            # Trade the seed for its equivalent generator in order to be able to restore the state.
            generator = torch.Generator(device=device).manual_seed(call_kwargs["seed"])
            del call_kwargs["seed"]
            call_kwargs["generator"] = generator

            steps_completed = 0
            img_list = []

            if latest_checkpoint is not None:
                with core_context.checkpoint.restore_path(latest_checkpoint) as path:
                    with open(path.joinpath("metadata.json"), "r") as f:
                        steps_completed = json.load(f)["steps_completed"]
                        generator_state_dict = torch.load(
                            path.joinpath("generator_state_dict.pt"),
                            map_location=device,
                        )
                        generator.set_state(generator_state_dict[device])

            # There will be a single op of len max_length, as defined in the searcher config.
            for op in core_context.searcher.operations():
                while steps_completed < op.length:
                    # Ensure all workers are using different, not-previously-used seeds.
                    img_list.extend(pipeline(**call_kwargs))
                    steps_completed += 1

                    # Write to tensorboard at the specified frequency.
                    if steps_completed % save_freq == 0 or steps_completed == op.length:
                        # Gather all and images to the main process
                        tags_and_imgs = core_context.distributed.gather((tb_tag, img_list))
                        devices_and_generators = core_context.distributed.gather(
                            (device, generator.get_state())
                        )
                        if is_main_process:
                            # Upload images to tensorboard.
                            for tag, img_list in tags_and_imgs:
                                img_ts = torch.stack(
                                    [pil_to_tensor(img) for img in img_list], dim=0
                                )
                                tb_writer.add_images(
                                    tag,
                                    img_tensor=img_ts,
                                    global_step=steps_completed,
                                )
                                tb_writer.flush()  # Ensure all images are written to disk.
                                core_context.train.upload_tensorboard_files()
                            # Save the state of the generators as the checkpoint.
                            checkpoint_metadata_dict = {
                                "steps_completed": steps_completed,
                            }
                            with core_context.checkpoint.store_path(checkpoint_metadata_dict) as (
                                path,
                                storage_id,
                            ):
                                generator_state_dict = {
                                    device: state for device, state in devices_and_generators
                                }
                                torch.save(
                                    generator_state_dict, path.joinpath("generator_state_dict.pt")
                                )
                            op.report_progress(steps_completed)
                        # Reset image list.
                        img_list = []
                    if core_context.preempt.should_preempt():
                        return

                if is_main_process:
                    # Report zero upon completion.
                    op.report_completed(0)

    def load_from_checkpoint_paths(
        self, checkpoint_paths: Union[Union[str, pathlib.Path], List[Union[str, pathlib.Path]]]
    ) -> None:
        """Load concepts from one or more checkpoint paths, each of which is expected contain a
        file with the name matching the `learned_embeddings_filename` init arg. The file is
        expected to contain a dictionary whose keys are the `concept_token`s and whose values are
        dictionaries containing an `initializer_token` key and a `learned_embeddings` whose
        corresponding values are the initializer string and learned embedding tensors, respectively.
        """
        if not checkpoint_paths:
            return
        # Get data from all checkpoints.
        if isinstance(checkpoint_paths, str):
            checkpoint_paths = [pathlib.Path(checkpoint_paths)]
        if isinstance(checkpoint_paths, pathlib.Path):
            checkpoint_paths = [checkpoint_paths]

        for path in checkpoint_paths:
            if isinstance(path, str):
                path = pathlib.Path(path)
            # TODO: Check that the same pretrained_model_name_or_path is used for all ckpts.
            learned_embeddings_dict = torch.load(path.joinpath(self.learned_embeddings_filename))
            # Update embedding matrix and attrs.
            for concept_token, embedding_dict in learned_embeddings_dict.items():
                if concept_token in self.learned_embeddings_dict:
                    raise ValueError(
                        f"Checkpoint concept conflict: {concept_token} already exists."
                    )
                initializer_tokens = embedding_dict["initializer_tokens"]
                learned_embeddings = embedding_dict["learned_embeddings"]
                (
                    initializer_ids,
                    dummy_placeholder_ids,
                    dummy_placeholder_tokens,
                ) = utils.add_new_tokens_to_tokenizer(
                    concept_token=concept_token,
                    initializer_tokens=initializer_tokens,
                    tokenizer=self.tokenizer,
                )

                self.text_encoder.resize_token_embeddings(len(self.tokenizer))
                token_embeddings = self.text_encoder.get_input_embeddings().weight.data
                # Sanity check on length.
                # TODO: replace with strict=True in zip after upgrade to py >= 3.10
                assert len(dummy_placeholder_ids) == len(
                    learned_embeddings
                ), "dummy_placeholder_ids and learned_embeddings must have the same length"
                for d_id, tensor in zip(dummy_placeholder_ids, learned_embeddings):
                    token_embeddings[d_id] = tensor
                self.learned_embeddings_dict[concept_token] = embedding_dict
                self.all_added_concepts.append(concept_token)
                self.concept_to_dummy_tokens_map[concept_token] = dummy_placeholder_tokens
            self.all_checkpoint_paths.append(path)

        print(f"Successfully loaded checkpoints. All loaded concepts: {self.all_added_concepts}")

    def load_from_uuids(
        self,
        uuids: Union[str, Sequence[str]],
    ) -> None:
        """Load concepts from one or more Determined checkpoint uuids. Must be logged into the
        Determined cluster to use this method.  If not logged-in, call
        determined.experimental.client.login first.
        """
        if isinstance(uuids, str):
            uuids = [uuids]
        checkpoint_paths = []
        for u in uuids:
            checkpoint = client.get_checkpoint(u)
            checkpoint_paths.append(pathlib.Path(checkpoint.download()))
        self.load_from_checkpoint_paths(checkpoint_paths)

    def _build_models(self) -> None:
        print(80 * "-", "Downloading pre-trained models...", 80 * "-", sep="\n")
        revision = "fp16" if self.use_fp16 else "main"
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            subfolder="tokenizer",
            use_auth_token=self.use_auth_token,
            revision=revision,
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            subfolder="text_encoder",
            use_auth_token=self.use_auth_token,
            revision=revision,
        )
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            subfolder="vae",
            use_auth_token=self.use_auth_token,
            revision=revision,
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            subfolder="unet",
            use_auth_token=self.use_auth_token,
            revision=revision,
        )
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            pretrained_model_name_or_path="CompVis/stable-diffusion-safety-checker"
        )
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained(
            pretrained_model_name_or_path="openai/clip-vit-base-patch32"
        )

        for model in (
            self.text_encoder,
            self.vae,
            self.unet,
        ):
            model.to(self.device)
            model.eval()

    def _build_pipeline(self) -> None:
        print(
            80 * "-",
            "Building the pipeline...",
            80 * "-",
            sep="\n",
        )
        self.text_encoder.eval()
        self.pipeline = StableDiffusionPipeline(
            text_encoder=self.text_encoder,
            vae=self.vae,
            unet=self.unet,
            tokenizer=self.tokenizer,
            scheduler=self.scheduler,
            safety_checker=self.safety_checker,
            feature_extractor=self.feature_extractor,
        ).to(self.device)
        print("Done!")

    def _replace_concepts_with_dummies(self, text: str) -> str:
        for concept_token, dummy_tokens in self.concept_to_dummy_tokens_map.items():
            text = text.replace(concept_token, dummy_tokens)
        return text

    def _save_img(self, img: Image.Image, filename: str, saved_img_dir: str) -> None:
        saved_img_dir = pathlib.Path(saved_img_dir)
        save_path = saved_img_dir.joinpath(filename)
        img.save(save_path)

    def __call__(
        self,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: int = 7.5,
        saved_img_dir: Optional[str] = None,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        num_samples: int = 1,
        batch_size: int = 1,
        other_hf_pipeline_call_kwargs: Optional[dict] = None,
        max_nsfw_retries: Optional[int] = 5,
    ) -> List[Image.Image]:
        """Generates a list of num_samples images from the provided prompt and optionally writes the
        results to disk.
        """
        assert not (
            seed is not None and generator is not None
        ), "Only one of `seed` or `generator` can be provided."
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        other_hf_pipeline_call_kwargs = other_hf_pipeline_call_kwargs or {}
        imgs = []
        generated_samples = retries = 0
        # The dummy prompts are what actually get fed into the pipeline
        dummy_prompt = self._replace_concepts_with_dummies(prompt)
        while generated_samples < num_samples:
            context = torch.autocast("cuda") if self.use_autocast else nullcontext()
            with context:
                output = self.pipeline(
                    [dummy_prompt] * batch_size,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    **other_hf_pipeline_call_kwargs,
                )
            for nsfw, img in zip(output["nsfw_content_detected"], output["sample"]):
                if nsfw and retries < max_nsfw_retries:
                    retries += 1
                    continue
                imgs.append(img)
                generated_samples += 1
                if generated_samples == num_samples:
                    break

        if saved_img_dir is not None:
            for idx, img in enumerate(imgs):
                file_ending = (
                    f"_{num_inference_steps}_steps_{guidance_scale}_gs_{seed}_seed_{idx}.png"
                )
                joined_split_prompt = "_".join(prompt.split())
                filename = joined_split_prompt[: 255 - len(file_ending)] + file_ending
                self._save_img(img, filename, saved_img_dir)

        return imgs

    def __repr__(self) -> str:
        attr_dict = {
            "scheduler_name": self.scheduler_name,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "beta_schedule": self.beta_schedule,
            "other_scheduler_kwargs": self.other_scheduler_kwargs,
            "pretrained_model_name_or_path": self.pretrained_model_name_or_path,
            "device": self.device,
            "use_autocast": self.use_autocast,
            "use_fp16": self.use_fp16,
            "all_added_concepts": self.all_added_concepts,
        }
        attr_dict_str = ", ".join([f"{key}={value}" for key, value in attr_dict.items()])
        return f"{self.__class__.__name__}({attr_dict_str})"


def add_new_tokens_to_tokenizer(
    concept_token: str,
    initializer_tokens: Sequence[str],
    tokenizer: nn.Module,
) -> Tuple[List[int], List[int], str]:
    """Helper function for adding new tokens to the tokenizer and extending the corresponding
    embeddings appropriately, given a single concept token and its sequence of corresponding
    initializer tokens.  Returns the lists of ids for the initializer tokens and their dummy
    replacements, as well as the string representation of the dummies.
    """
    initializer_ids = tokenizer(
        initializer_tokens,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids

    try:
        special_token_ids = tokenizer.all_special_ids
    except AttributeError:
        special_token_ids = []

    non_special_initializer_locations = torch.isin(
        initializer_ids, torch.tensor(special_token_ids), invert=True
    )
    non_special_initializer_ids = initializer_ids[non_special_initializer_locations]
    if len(non_special_initializer_ids) == 0:
        raise ValueError(
            f'"{initializer_tokens}" maps to trivial tokens, please choose a different initializer.'
        )

    # Add a dummy placeholder token for every token in the initializer.
    dummy_placeholder_token_list = [
        f"{concept_token}_{n}" for n in range(len(non_special_initializer_ids))
    ]
    dummy_placeholder_tokens = " ".join(dummy_placeholder_token_list)
    num_added_tokens = tokenizer.add_tokens(dummy_placeholder_token_list)
    if num_added_tokens != len(dummy_placeholder_token_list):
        raise ValueError(
            f"Subset of {dummy_placeholder_token_list} tokens already exist in tokenizer."
        )

    dummy_placeholder_ids = tokenizer.convert_tokens_to_ids(dummy_placeholder_token_list)
    # Sanity check
    assert len(dummy_placeholder_ids) == len(
        non_special_initializer_ids
    ), 'Length of "dummy_placeholder_ids" and "non_special_initializer_ids" must match.'

    return non_special_initializer_ids, dummy_placeholder_ids, dummy_placeholder_tokens