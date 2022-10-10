import json
import logging
import os
import pathlib
from PIL import Image
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import determined as det
import torch
from determined.experimental import client
from diffusers import (
    StableDiffusionPipeline,
)
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import pil_to_tensor

from detsd import utils, defaults


class DetSDTextualInversionPipeline:
    """Class for generating images from a Stable Diffusion checkpoint trained using Determined
    AI. Initialize with no arguments in order to run plan Stable Diffusion without any trained
    textual inversion embeddings. Can optionally be run on a Determined cluster through the
    .generate_on_cluster() method for large-scale generation. Only intended for use with a GPU.
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
        use_fp16: bool = True,
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
            other_scheduler_kwargs or defaults.DEFAULT_SCHEDULER_KWARGS_DICT[scheduler_name]
        )
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.device = device
        self.use_fp16 = use_fp16

        scheduler_kwargs = {
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "beta_schedule": self.beta_schedule,
            **self.other_scheduler_kwargs,
        }
        self.scheduler = defaults.NOISE_SCHEDULER_DICT[self.scheduler_name](**scheduler_kwargs)

        # The below attrs are non-trivially instantiated as necessary through appropriate methods.
        self.all_checkpoint_paths = []
        self.learned_embeddings_dict = {}
        self.concept_to_dummy_tokens_map = {}
        self.all_added_concepts = []

        self._build_pipeline()

    @classmethod
    def generate_on_cluster(cls) -> None:
        """Creates a DetSDTextualInversionPipeline instance on the cluster, drawing hyperparameters
        and other needed information from the Determined master, and then generates images. Expects
        the `hyperparameters` section of the config to be broken into the following sections:
        - `pipeline`: containing all __init__ args
        - `uuids`: a (possibly empty) array of any checkpoint UUIDs which are to be loaded into
        the pipeline.
        - `call`: all arguments which are to be passed to the `__call__` method which generates
        images, but which cannot contain `seed` or `generator` or `max_nsfw_retries` for efficiency
        and reproducibility reasons.
        - `main_process_seed`: an integer specifying the seed used by the chief worker for
        generation. Other workers add their process index to this value.
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
        main_process_seed = hparams["main_process_seed"]
        save_freq = hparams["save_freq"]

        assert not call_kwargs.get(
            "max_nsfw_retries", 0
        ), "max_nsfw_retries must be 0 when using generate_on_cluster()."

        logger = logging.getLogger(__name__)

        # Get the distributed context, as needed.
        try:
            distributed = det.core.DistributedContext.from_torch_distributed()
        except KeyError:
            distributed = None

        with det.core.init(
            distributed=distributed, tensorboard_mode=det.core.TensorboardMode.MANUAL
        ) as core_context:
            # Get worker data.
            process_index = core_context.distributed.get_rank()
            is_main_process = process_index == 0
            local_process_index = core_context.distributed.get_local_rank()
            is_local_main_process = local_process_index == 0

            device = f"cuda:{process_index}" if distributed is not None else "cuda"
            pipeline_init_kwargs["device"] = device

            # Instantiate the pipeline and load in any checkpoints by uuid.
            pipeline = cls(**pipeline_init_kwargs)
            # Only the local chief worker performs the download.
            if is_local_main_process:
                paths = pipeline.load_from_uuids(uuid_list)
            else:
                paths = None
            paths = core_context.distributed.broadcast_local(paths)
            if not is_local_main_process:
                pipeline.load_from_checkpoint_paths(paths)

            # Create the Tensorboard writer.
            tb_dir = core_context.train.get_tensorboard_path()
            tb_writer = SummaryWriter(log_dir=tb_dir)
            # Include the __call__ args in the tensorboard tag.
            tb_tag = ", ".join([f"{k}: {v}" for k, v in call_kwargs.items() if v])

            # Use unique seeds, to avoid repeated images, and add the corresponding generator to the
            # call_kwargs.
            seed = main_process_seed + process_index
            generator = torch.Generator(device=device).manual_seed(seed)
            call_kwargs["generator"] = generator
            # Add seed information to the tensorboard tag.
            tb_tag += f", seed: {seed}"

            steps_completed = 0
            generated_imgs = 0
            img_history = []

            if latest_checkpoint is not None:
                with core_context.checkpoint.restore_path(latest_checkpoint) as path:
                    with open(path.joinpath("metadata.json"), "r") as f:
                        metadata_dict = json.load(f)
                        generator_state_dict = torch.load(
                            path.joinpath("generator_state_dict.pt"),
                        )
                        steps_completed = metadata_dict["steps_completed"]
                        generated_imgs = metadata_dict["generated_imgs"]
                        generator.set_state(generator_state_dict[device])
                if is_main_process:
                    logger.info(f"Resumed from checkpoint at step {steps_completed}")

            if is_main_process:
                logger.info("--------------- Generating Images ---------------")

            # There will be a single op of len max_length, as defined in the searcher config.
            for op in core_context.searcher.operations():
                while steps_completed < op.length:
                    img_history.extend(pipeline(**call_kwargs))
                    steps_completed += 1

                    # Write to tensorboard and checkpoint at the specified frequency.
                    if steps_completed % save_freq == 0 or steps_completed == op.length:
                        # Gather all and images to the main process
                        tags_and_imgs = core_context.distributed.gather((tb_tag, img_history))
                        devices_and_generators = core_context.distributed.gather(
                            (device, generator.get_state())
                        )
                        if is_main_process:
                            logger.info(f"Saving at step {steps_completed}")
                            print("tags_and_imgs", tags_and_imgs)
                            for tag, img_list in tags_and_imgs:
                                print(
                                    "TEST",
                                    "tag",
                                    tag,
                                    "img_list",
                                    img_list,
                                    "generated_imgs",
                                    generated_imgs,
                                )
                                for idx, img in enumerate(img_list):
                                    img_t = pil_to_tensor(img)
                                    global_step = generated_imgs + idx
                                    print("global_step", global_step)
                                    tb_writer.add_image(
                                        tag,
                                        img_tensor=img_t,
                                        global_step=global_step,
                                    )
                                tb_writer.flush()  # Ensure all images are written to disk.
                                core_context.train.upload_tensorboard_files()
                            # Save the state of the generators as the checkpoint.
                            generated_imgs += len(img_history)
                            checkpoint_metadata_dict = {
                                "steps_completed": steps_completed,
                                "generated_imgs": generated_imgs,
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
                        img_history = []
                        # Only preempt after a checkpoint has been saved.
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
                    tokenizer=self.pipeline.tokenizer,
                )

                self.pipeline.text_encoder.resize_token_embeddings(len(self.pipeline.tokenizer))
                token_embeddings = self.pipeline.text_encoder.get_input_embeddings().weight.data
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

    def load_from_uuids(
        self,
        uuids: Union[str, Sequence[str]],
    ) -> List[pathlib.Path]:
        """Load concepts from one or more Determined checkpoint uuids and returns a list of all
        downloaded checkpoint paths. Must be logged into the Determined cluster to use this method.
        If not logged-in, call determined.experimental.client.login first.
        """
        if isinstance(uuids, str):
            uuids = [uuids]
        checkpoint_paths = []
        for u in uuids:
            checkpoint = client.get_checkpoint(u)
            checkpoint_paths.append(pathlib.Path(checkpoint.download()))
        self.load_from_checkpoint_paths(checkpoint_paths)
        return checkpoint_paths

    def _build_pipeline(self) -> None:
        revision = "fp16" if self.use_fp16 else "main"
        torch_dtype = torch.float16 if self.use_fp16 else None
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            scheduler=self.scheduler,
            use_auth_token=self.use_auth_token,
            revision=revision,
            torch_dtype=torch_dtype,
        ).to(self.device)

    def _replace_concepts_with_dummies(self, text: str) -> str:
        for concept_token, dummy_tokens in self.concept_to_dummy_tokens_map.items():
            text = text.replace(concept_token, dummy_tokens)
        return text

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
        max_nsfw_retries: int = 0,
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
                call_args_str = f"_{num_inference_steps}_steps_{guidance_scale}_gs"
                if seed is not None:
                    call_args_str += f"_seed_{seed}"
                elif generator is not None:
                    call_args_str += f"_seed_{generator.initial_seed()}"
                file_ending = call_args_str + ".png"
                joined_split_prompt = "_".join(prompt.split())
                filename = joined_split_prompt[: 255 - len(file_ending)] + file_ending
                save_path = pathlib.Path(saved_img_dir).joinpath(filename)
                img.save(save_path)

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
            "use_fp16": self.use_fp16,
            "all_added_concepts": self.all_added_concepts,
        }
        attr_dict_str = ", ".join([f"{key}={value}" for key, value in attr_dict.items()])
        return f"{self.__class__.__name__}({attr_dict_str})"
