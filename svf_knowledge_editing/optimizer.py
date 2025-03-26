import abc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from logging_utils import get_mean_std_max_min_dict
from utils import (backward, eval_model, forward, load_base_params,
                   load_hf_params_to_vllm)


class OptimizationAlgorithm(abc.ABC):
    def __init__(self, **kwargs):
        nn.Module.__init__(self=self)

    @abc.abstractmethod
    def step_optimization(
        self,
        model_id,
        model,
        tokenizer,
        policy,
        task_loader,
        batch_ix,
        train_data,
        train_eval,
        base_params,
        decomposed_params,
        original_model_params,
        metrics_to_log,
        vllm_model=None,
        **kwargs,
    ):
        raise NotADirectoryError

    @abc.abstractmethod
    def update(self, policy):
        raise NotImplementedError

    def log_optim(self, metrics_to_log):
        pass


class Reinforce(OptimizationAlgorithm, nn.Module):
    def __init__(
        self, policy, gpu, max_grad_norm, lr, rw_norm, rw_clip, kl_ref_coeff, **kwargs
    ):
        nn.Module.__init__(self=self)
        self.gpu = gpu
        self.kl_ref_coeff = kl_ref_coeff
        self.use_kl_loss = kl_ref_coeff > 0.0
        self.max_grad_norm = float(max_grad_norm)
        self.lr = lr
        self.rw_norm = rw_norm
        self.rw_clip = rw_clip
        self.optimizer = torch.optim.Adam(policy.trainable_params, lr=lr)

    def compute_ref_logprobs(
        self,
        model,
        tokenizer,
        prompts,
        res,
    ):
        ref_log_probs_list = []
        print("Computing reference log probs...")
        for j, prompt in enumerate(prompts):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(self.gpu)
            prompt_length = input_ids.shape[-1]
            output_ids = tokenizer(
                prompt + res.sample_details[j]["target_new"],
                return_tensors="pt",
            ).input_ids.to(self.gpu)
            outputs = model(output_ids)
            logits = outputs.logits[:, prompt_length - 1 : -1]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            ref_log_probs_list.append(log_probs.detach().cpu())
        return ref_log_probs_list

    def get_rewards(self, task_loader, res):
        rw_norm = self.rw_norm
        rw_clip = self.rw_clip
        rewards = task_loader.get_rewards(res=res)

        if rw_norm:
            rewards = np.array(rewards)
            mean_rw = np.mean(rewards)
            std_rw = np.clip(np.std(rewards), a_min=1e-7, a_max=None)
            rewards = (rewards - mean_rw) / std_rw
        if rw_clip is not None:
            if rw_clip > 0:
                rewards = np.array(rewards)
                rewards = np.clip(rewards, a_min=-rw_clip, a_max=rw_clip)
        return rewards

    def step_optimization(
        self,
        model_id,
        model,
        tokenizer,
        policy,
        task_loader,
        batch_ix,
        train_data,
        train_eval,
        base_params,
        decomposed_params,
        original_model_params,
        metrics_to_log,
        vllm_model=None,
        **kwargs,
    ):
        use_kl_loss = self.use_kl_loss
        kl_ref_coeff = self.kl_ref_coeff

        gpu = self.gpu

        prompts = [
            task_loader.get_prompt(
                tokenizer,
                train_data,
                i,
                model_id=model_id,
            )
            for i in batch_ix
        ]

        clipped_batch_size = len(prompts)

        learnable_params = policy.get_learnable_params()
        new_params = forward(
            policy, model, base_params, decomposed_params, learnable_params
        )

        print("Loading weights and getting completions with VLLM")
        load_hf_params_to_vllm(new_params, vllm_model.llm)
        res = eval_model(vllm_model, train_eval, batch_ix)
        rewards = self.get_rewards(task_loader=task_loader, res=res)

        rw_stats = get_mean_std_max_min_dict(array=rewards, prefix="rewards")
        metrics_to_log.update(**rw_stats)

        if use_kl_loss:
            with torch.no_grad():
                load_base_params(model=model, base_params=original_model_params)
                ref_log_probs_list = self.compute_ref_logprobs(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    res=res,
                )
                new_params = forward(
                    policy, model, base_params, decomposed_params, learnable_params
                )

        print("Computing the policy gradient...")
        for j, prompt in enumerate(prompts):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(gpu)
            prompt_length = input_ids.shape[-1]
            output_ids = tokenizer(
                prompt + res.sample_details[j]["target_new"],
                return_tensors="pt",
            ).input_ids.to(gpu)
            generated_ids = output_ids[:, prompt_length:]

            outputs = model(output_ids)
            logits = outputs.logits[:, prompt_length - 1 : -1]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs.gather(
                2, generated_ids.unsqueeze(-1)
            ).squeeze(-1)
            log_likelihood = selected_log_probs.sum(axis=-1)

            pg = -log_likelihood * rewards[j]
            loss = pg

            if use_kl_loss:
                ref_log_probs = ref_log_probs_list[j].to(gpu)
                kl_div = F.kl_div(
                    input=log_probs,
                    target=ref_log_probs,
                    log_target=True,
                    reduction="sum",
                )
                loss = loss + kl_ref_coeff * kl_div
            scaled_loss = loss / clipped_batch_size
            scaled_loss.backward()
            log_dict = {
                "pg": pg.item(),
                "loss": loss.item(),
            }
            if use_kl_loss:
                log_dict["kl_div"] = kl_div.item()
            metrics_to_log.update(**log_dict)
        backward(policy, model, base_params, decomposed_params, learnable_params)

    def update(self, policy):
        max_grad_norm = self.max_grad_norm
        torch.nn.utils.clip_grad_norm_(policy.trainable_params, max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def log_optim(self, metrics_to_log):
        metrics_dict = metrics_to_log.get()
        pg = metrics_dict["pg"]
        print(f"PG={pg}")
        if self.use_kl_loss:
            kl_div = metrics_dict["kl_div"]
            print(f"kl_div={kl_div}")
