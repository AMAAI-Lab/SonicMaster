from transformers import T5EncoderModel, T5TokenizerFast
import torch
from diffusers import FluxTransformer2DModel
from torch import nn
import random
from typing import List
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling
import copy
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from typing import Optional, Union, List
from datasets import load_dataset, Audio
from math import pi
import inspect
import yaml
from utils import sample_linear_plus_uniform

def rk4_step(latents, t, dt, model, batch_size, encoder_hidden_states, pooled_projection, txt_ids, audio_ids, device, classifier_free_guidance=False, guidance_scale=1):
    def velocity_fn(latents, t_input, classifier_free_guidance, guidance_scale=1):
        t_batched = t_input.expand(batch_size).to(device)

        x = (
            torch.cat([latents] * 2) if classifier_free_guidance else latents
        )
        # x_in = x.repeat(2, 1, 1)
        # t_in = t_batched.repeat(2)
        # pooled_proj_in = pooled_projection.repeat(2, 1)
        # encoder_hidden_states_in = encoder_hidden_states.repeat(2, 1, 1)


        # txt_ids_in = torch.cat([
        #     torch.zeros_like(txt_ids),  # unconditional (empty prompt)
        #     txt_ids
        # ], dim=0)
        # audio_ids_in = torch.cat([
        #     torch.zeros_like(audio_ids),  # optional if needed
        #     audio_ids
        # ], dim=0)

        velo_pred = model(
            hidden_states=x,
            timestep=t_batched,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projection,
            txt_ids=txt_ids,
            img_ids=audio_ids,
            guidance=None,
            return_dict=False,
        )[0]

        if classifier_free_guidance:
            velo_pred_uncond, velo_pred_text = velo_pred.chunk(2)
            velo_out = velo_pred_uncond + guidance_scale * (velo_pred_text - velo_pred_uncond)
            return velo_out

        else:
            return velo_pred


    k1 = velocity_fn(latents, t, classifier_free_guidance, guidance_scale)
    k2 = velocity_fn(latents + 0.5 * dt * k1, t - 0.5 * dt, classifier_free_guidance, guidance_scale)
    k3 = velocity_fn(latents + 0.5 * dt * k2, t - 0.5 * dt, classifier_free_guidance, guidance_scale)
    k4 = velocity_fn(latents + dt * k3, t - dt, classifier_free_guidance, guidance_scale)

    latents_next = latents + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return latents_next



class StableAudioPositionalEmbedding(nn.Module):
    """Used for continuous time
    Adapted from Stable Audio Open.
    """

    def __init__(self, dim: int):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        times = times[..., None]
        freqs = times * self.weights[None] * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((times, fouriered), dim=-1)
        return fouriered


class DurationEmbedder(nn.Module):
    """
    A simple linear projection model to map numbers to a latent space.

    Code is adapted from
    https://github.com/Stability-AI/stable-audio-tools

    Args:
        number_embedding_dim (`int`):
            Dimensionality of the number embeddings.
        min_value (`int`):
            The minimum value of the seconds number conditioning modules.
        max_value (`int`):
            The maximum value of the seconds number conditioning modules
        internal_dim (`int`):
            Dimensionality of the intermediate number hidden states.
    """

    def __init__(
        self,
        number_embedding_dim,
        min_value,
        max_value,
        internal_dim: Optional[int] = 256,
    ):
        super().__init__()
        self.time_positional_embedding = nn.Sequential(
            StableAudioPositionalEmbedding(internal_dim),
            nn.Linear(in_features=internal_dim + 1, out_features=number_embedding_dim),
        )

        self.number_embedding_dim = number_embedding_dim
        self.min_value = min_value
        self.max_value = max_value
        self.dtype = torch.float32

    def forward(
        self,
        floats: torch.Tensor,
    ):
        floats = floats.clamp(self.min_value, self.max_value)

        normalized_floats = (floats - self.min_value) / (
            self.max_value - self.min_value
        )

        # Cast floats to same type as embedder
        embedder_dtype = next(self.time_positional_embedding.parameters()).dtype
        normalized_floats = normalized_floats.to(embedder_dtype)

        embedding = self.time_positional_embedding(normalized_floats)
        float_embeds = embedding.view(-1, 1, self.number_embedding_dim)

        return float_embeds


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):

    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class TangoFlux(nn.Module):

    def __init__(self, config, text_encoder_dir=None, initialize_reference_model=False,):

        super().__init__()

        self.num_layers = config.get("num_layers", 6)
        self.num_single_layers = config.get("num_single_layers", 18)
        self.in_channels = config.get("in_channels", 64)
        self.attention_head_dim = config.get("attention_head_dim", 128)
        self.joint_attention_dim = config.get("joint_attention_dim", 1024)
        self.num_attention_heads = config.get("num_attention_heads", 8)
        self.audio_seq_len = config.get("audio_seq_len", 645)
        self.max_duration = config.get("max_duration", 30)
        self.uncondition = config.get("uncondition", False)
        self.text_encoder_name = config.get("text_encoder_name", "google/flan-t5-large")

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)
        self.max_text_seq_len = 64
        self.text_encoder = T5EncoderModel.from_pretrained(
            text_encoder_dir if text_encoder_dir is not None else self.text_encoder_name
        )
        self.tokenizer = T5TokenizerFast.from_pretrained(
            text_encoder_dir if text_encoder_dir is not None else self.text_encoder_name
        )
        self.text_embedding_dim = self.text_encoder.config.d_model

        self.fc_text = nn.Sequential(
            nn.Linear(self.text_embedding_dim, self.joint_attention_dim), nn.ReLU()
        )
        self.fc_text_audio = nn.Sequential(
            nn.Linear(2*self.text_embedding_dim, self.joint_attention_dim), nn.ReLU()
        )
        self.audio_cond = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=7, stride=2, padding=3), #21 receptive field = roughly 1 sec
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(output_size=1),  # Output shape: [B, 512, 1]
            nn.Flatten(),                         # Output shape: [B, 512]

            nn.Linear(in_features=512, out_features=1024),  # Final projection to match text dim
            nn.ReLU(),
        )

        self.duration_emebdder = DurationEmbedder(
            self.text_embedding_dim, min_value=0, max_value=self.max_duration
        )

        self.transformer = FluxTransformer2DModel(
            in_channels=self.in_channels,
            num_layers=self.num_layers,
            num_single_layers=self.num_single_layers,
            attention_head_dim=self.attention_head_dim,
            num_attention_heads=self.num_attention_heads,
            joint_attention_dim=self.joint_attention_dim,
            pooled_projection_dim=self.text_embedding_dim,
            guidance_embeds=False,
        )

        self.beta_dpo = 2000  ## this is used for dpo training

    def get_sigmas(self, timesteps, n_dim=3, dtype=torch.float32):
        device = self.text_encoder.device
        sigmas = self.noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)

        schedule_timesteps = self.noise_scheduler_copy.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def encode_text_classifier_free(self, prompt: List[str], num_samples_per_prompt=1):
        device = self.text_encoder.device
        batch = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(
            device
        )

        with torch.no_grad():
            prompt_embeds = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )[0]

        prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        attention_mask = attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        # get unconditional embeddings for classifier free guidance
        uncond_tokens = [""]

        max_length = prompt_embeds.shape[1]
        uncond_batch = self.tokenizer(
            uncond_tokens,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        uncond_input_ids = uncond_batch.input_ids.to(device)
        uncond_attention_mask = uncond_batch.attention_mask.to(device)

        with torch.no_grad():
            negative_prompt_embeds = self.text_encoder(
                input_ids=uncond_input_ids, attention_mask=uncond_attention_mask
            )[0]

        negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(
            num_samples_per_prompt, 0
        ) #these two only have batch 1 - would need to expand to match batch size... then concat
        uncond_attention_mask = uncond_attention_mask.repeat_interleave(
            num_samples_per_prompt, 0
        )

        # For classifier free guidance, we need to do two forward passes.
        # We concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_mask = torch.cat([uncond_attention_mask, attention_mask])
        boolean_prompt_mask = (prompt_mask == 1).to(device)

        return prompt_embeds, boolean_prompt_mask

    @torch.no_grad()
    def encode_text(self, prompt):
        device = self.text_encoder.device
        batch = self.tokenizer(
            prompt,
            max_length=self.max_text_seq_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(
            device
        )

        encoder_hidden_states = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )[0]

        boolean_encoder_mask = (attention_mask == 1).to(device)

        return encoder_hidden_states, boolean_encoder_mask

    def encode_duration(self, duration):
        return self.duration_emebdder(duration)

    @torch.no_grad()
    def inference_flow(
        self,
        latents,
        prompt,
        audiocond_latents=None,
        num_inference_steps=50,
        timesteps=None,
        guidance_scale=3,
        duration=10,
        seed=0,
        disable_progress=False,
        num_samples_per_prompt=1,
        callback_on_step_end=None,
        solver="Euler", #or rk4
    ):
        """Only tested for single inference. Haven't test for batch inference"""
        
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

        bsz = num_samples_per_prompt
        device = self.transformer.device
        audio_seq_length = self.audio_seq_len
        # scheduler = self.noise_scheduler

        # if audiocond_latents==None:
        #     audiocond_latents=torch.zeros_like(latents)

        if not isinstance(prompt, list):
            prompt = [prompt]
        if not isinstance(duration, torch.Tensor):
            duration = torch.tensor([duration], device=device) #make this batch?
        classifier_free_guidance = guidance_scale > 1.0
        duration_hidden_states = self.encode_duration(duration)
        if classifier_free_guidance:
            bsz = 2 * num_samples_per_prompt

            encoder_hidden_states, boolean_encoder_mask = (
                self.encode_text_classifier_free(
                    prompt, num_samples_per_prompt=num_samples_per_prompt
                )
            )
            duration_hidden_states = duration_hidden_states.repeat(bsz, 1, 1)

        else:

            encoder_hidden_states, boolean_encoder_mask = self.encode_text(
                prompt
            )

        mask_expanded = boolean_encoder_mask.unsqueeze(-1).expand_as(
            encoder_hidden_states
        )
        masked_data = torch.where(
            mask_expanded, encoder_hidden_states, torch.tensor(float("nan"))
        )

        # pooled = torch.nanmean(masked_data, dim=1)
        # pooled_projection = self.fc(pooled)


        pooled = torch.nanmean(masked_data, dim=1) # text part of pooling
        pooled_projection_text = self.fc_text(pooled)

        if audiocond_latents==None:
            pooled_projection_audio=torch.zeros_like(pooled_projection_text)
        else:
            pooled_projection_audio = self.audio_cond(audiocond_latents[:,:audio_seq_length // 3,:].transpose(1,2)) # audio part of pooling, input [B,64,T]
            if classifier_free_guidance: #CFG stack zeros with condition
                pooled_projection_audio=torch.cat([torch.zeros_like(pooled_projection_audio),pooled_projection_audio],dim=0) #CFG
        # print(pooled_projection_text.shape,pooled_projection_audio.shape)
        text_audio_cat = torch.cat([pooled_projection_text, pooled_projection_audio], dim=1)

        pooled_projection = self.fc_text_audio(text_audio_cat)


        encoder_hidden_states = torch.cat(
            [encoder_hidden_states, duration_hidden_states], dim=1
        )  ## (bs,seq_len,dim)

        # sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

        # timesteps = torch.linspace(0.0,1.0,num_inference_steps+1)
        # dt = timesteps[1] - timesteps[0]
        # timesteps = timesteps[:-1] #so that we avoid taking a timestep of 1.0


        timesteps = torch.linspace(1.0,0.0,num_inference_steps+1)
        dt = timesteps[0] - timesteps[1]
        timesteps = timesteps[:-1] #so that we avoid taking a timestep of 1.0

        # timesteps, num_inference_steps = retrieve_timesteps(
        #     scheduler, num_inference_steps, device, timesteps, sigmas
        # )

        # latents = torch.randn(num_samples_per_prompt, self.audio_seq_len, 64)
        # weight_dtype = latents.dtype

        progress_bar = tqdm(range(num_inference_steps), disable=disable_progress)

        txt_ids = torch.zeros(bsz, encoder_hidden_states.shape[1], 3).to(device)
        audio_ids = (
            torch.arange(self.audio_seq_len)
            .unsqueeze(0)
            .unsqueeze(-1)
            .repeat(bsz, 1, 3)
            .to(device)
        )

        timesteps = timesteps.to(device)
        latents = latents.to(device)
        encoder_hidden_states = encoder_hidden_states.to(device)

        for i, t in enumerate(timesteps):

            if solver=="rk4":
                latents = rk4_step(latents, t, dt, self.transformer, bsz, encoder_hidden_states, pooled_projection, txt_ids, audio_ids, device, classifier_free_guidance, guidance_scale)
            else: #Euler
                latents_input = (
                    torch.cat([latents] * 2) if classifier_free_guidance else latents
                )

                velo_pred = self.transformer(
                    hidden_states=latents_input,
                    timestep=torch.tensor([t], device=device),
                    guidance=None,
                    pooled_projections=pooled_projection,
                    encoder_hidden_states=encoder_hidden_states,
                    txt_ids=txt_ids,
                    img_ids=audio_ids,
                    return_dict=False,
                )[0]

                if classifier_free_guidance:
                    velo_pred_uncond, velo_pred_text = velo_pred.chunk(2)
                    velo_pred= velo_pred_uncond + guidance_scale * (
                        velo_pred_text - velo_pred_uncond
                    )

                # latents = scheduler.step(velo_pred, t, latents).prev_sample #replace
                latents = latents + dt * velo_pred # Euler...

            progress_bar.update(1)

            if callback_on_step_end is not None:
                callback_on_step_end()

        return latents

    def forward(self, latents, deg_latents, prompt, duration=torch.tensor([10]), sft=True):

        device = latents.device
        audio_seq_length = self.audio_seq_len
        bsz = latents.shape[0]

        encoder_hidden_states, boolean_encoder_mask = self.encode_text(prompt)
        duration_hidden_states = self.encode_duration(duration)

        mask_expanded = boolean_encoder_mask.unsqueeze(-1).expand_as(
            encoder_hidden_states
        )
        masked_data = torch.where(
            mask_expanded, encoder_hidden_states, torch.tensor(float("nan"))
        )
        pooled = torch.nanmean(masked_data, dim=1) # text part of pooling
        pooled_projection_text = self.fc_text(pooled)
        
        audio_cond_input=latents[:,:audio_seq_length // 3,:]

        pooled_projection_audio = self.audio_cond(audio_cond_input.transpose(1,2)) # audio part of pooling, input [B,64,T]


        if self.training: #75% to drop the audio condition - we want the model to work with text primarily so it can get the first prediction very accurate
            mask = (torch.rand(pooled_projection_audio.size(0), device=pooled_projection_audio.device) > 0.75).float().unsqueeze(1)
            pooled_projection_audio = pooled_projection_audio * mask

        text_audio_cat = torch.cat([pooled_projection_text, pooled_projection_audio], dim=1)

        pooled_projection = self.fc_text_audio(text_audio_cat)

        # print("pooled_projection_text:", pooled_projection_text.shape)
        # print("pooled_projection_audio:", pooled_projection_audio.shape)
        # print("pooled_projection:", pooled_projection.shape)



        ## Add duration hidden states to encoder hidden states
        encoder_hidden_states = torch.cat(
            [encoder_hidden_states, duration_hidden_states], dim=1
        )  ## (bs,seq_len,dim)

        txt_ids = torch.zeros(bsz, encoder_hidden_states.shape[1], 3).to(device)
        audio_ids = (
            torch.arange(audio_seq_length)
            .unsqueeze(0)
            .unsqueeze(-1)
            .repeat(bsz, 1, 3)
            .to(device)
        )

        if self.uncondition:
            mask_indices = [k for k in range(len(prompt)) if random.random() < 0.1]
            if len(mask_indices) > 0:
                encoder_hidden_states[mask_indices] = 0


        u = sample_linear_plus_uniform(
            batch_size=bsz,
            skew_toward="start",          # Skew toward u = 0 -> more degrad audio
            uniform_weight=0.5,         # 1.5 at u=0 vs 0.5 at u=1
            device=latents.device
        )


        u_expanded = u.view(-1, 1, 1) #to match dimensions of the latents
        interpolated_input = (1.0 - u_expanded) * deg_latents + u_expanded * latents

        model_pred = self.transformer(
            hidden_states=interpolated_input,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projection,
            img_ids=audio_ids,
            txt_ids=txt_ids,
            guidance=None,
            timestep=1-u,
            return_dict=False,
        )[0]


        target = latents - deg_latents

        loss = torch.mean(
            ((model_pred.float() - target.float()) ** 2).reshape(
                target.shape[0], -1
            ),
            1,
        )
        loss = loss.mean()
        raw_model_loss, raw_ref_loss, implicit_acc = (
            0,
            0,
            0,
        )  ## default this to 0 if doing sft

        # else:
        #     encoder_hidden_states = encoder_hidden_states.repeat(2, 1, 1)
        #     pooled_projection = pooled_projection.repeat(2, 1)
        #     noise = (
        #         torch.randn_like(latents).chunk(2)[0].repeat(2, 1, 1)
        #     )  ## Have to sample same noise for preferred and rejected
        #     u = compute_density_for_timestep_sampling(
        #         weighting_scheme="logit_normal",
        #         batch_size=bsz // 2,
        #         logit_mean=0,
        #         logit_std=1,
        #         mode_scale=None,
        #     )

        #     indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()
        #     timesteps = self.noise_scheduler_copy.timesteps[indices].to(
        #         device=latents.device
        #     )
        #     timesteps = timesteps.repeat(2)
        #     sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)

        #     noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise

        #     model_pred = self.transformer(
        #         hidden_states=noisy_model_input,
        #         encoder_hidden_states=encoder_hidden_states,
        #         pooled_projections=pooled_projection,
        #         img_ids=audio_ids,
        #         txt_ids=txt_ids,
        #         guidance=None,
        #         # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
        #         timestep=timesteps / 1000,
        #         return_dict=False,
        #     )[0]
        #     target = noise - latents

        #     model_losses = F.mse_loss(
        #         model_pred.float(), target.float(), reduction="none"
        #     )
        #     model_losses = model_losses.mean(
        #         dim=list(range(1, len(model_losses.shape)))
        #     )
        #     model_losses_w, model_losses_l = model_losses.chunk(2)
        #     model_diff = model_losses_w - model_losses_l
        #     raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())

        #     with torch.no_grad():
        #         ref_preds = self.ref_transformer(
        #             hidden_states=noisy_model_input,
        #             encoder_hidden_states=encoder_hidden_states,
        #             pooled_projections=pooled_projection,
        #             img_ids=audio_ids,
        #             txt_ids=txt_ids,
        #             guidance=None,
        #             timestep=timesteps / 1000,
        #             return_dict=False,
        #         )[0]

        #         ref_loss = F.mse_loss(
        #             ref_preds.float(), target.float(), reduction="none"
        #         )
        #         ref_loss = ref_loss.mean(dim=list(range(1, len(ref_loss.shape))))

        #         ref_losses_w, ref_losses_l = ref_loss.chunk(2)
        #         ref_diff = ref_losses_w - ref_losses_l
        #         raw_ref_loss = ref_loss.mean()

        #     scale_term = -0.5 * self.beta_dpo
        #     inside_term = scale_term * (model_diff - ref_diff)
        #     implicit_acc = (
        #         scale_term * (model_diff - ref_diff) > 0
        #     ).sum().float() / inside_term.size(0)
        #     loss = -1 * F.logsigmoid(inside_term).mean() + model_losses_w.mean()

        # ## raw_model_loss, raw_ref_loss, implicit_acc is used to help to analyze dpo behaviour.
        return loss, raw_model_loss, raw_ref_loss, implicit_acc
