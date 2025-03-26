from abc import ABC, abstractmethod
import os

import vllm
import json
from transformers import AutoTokenizer
import math

from svf_knowledge_editing.eval.tasks.knowledge_editing import KnowledgeEditingTask
from svf_knowledge_editing.eval.models.vllm_model import VLLMModel, Message
from svf_knowledge_editing.prompts import KNOWLEDGE_ASSISTANT_PROMPT



def get_download_dir():
    if "HF_HOME" in os.environ:
        return os.environ["HF_HOME"] + "/models"
    else:
        return os.path.expanduser("~") + "/.cache/huggingface/models"


class Task(ABC):
    def __init__(
        self,
    ):
        self.model_to_template = {}
        self.system_msg = ()
        self.target_metric_train = None
        self.target_metric_valid = self.target_metric_train
        self.target_metric_test = self.target_metric_train
        self.target_metric_transfer = None
        self.has_transfer_split = True
        self.has_training_split = True

    @abstractmethod
    def get_train_data(
        self,
    ):
        raise NotImplementedError

    @abstractmethod
    def get_rewards(self, res):
        raise NotImplementedError

    @abstractmethod
    def get_evaluator(
        self,
        tokenizer: AutoTokenizer,
    ):
        raise NotImplementedError

    @abstractmethod
    def get_prompt(self, tokenizer, samples, ix, model_id):
        raise NotImplementedError

    @abstractmethod
    def get_vllm_model(self, model_id):
        raise NotImplementedError


class WikidataCounterfactTask(Task):
    def __init__(
        self, data_path,
    ):
        self.model_to_template = {
            "meta-llama/Llama-3.2-1B": (
                "{% set loop_messages = messages %}"
                "{% for message in loop_messages %}"
                "{% set content = ' ' + message['role'] + ': ' + message['content'] | trim + '\n' %}"
                "{% if loop.index0 == 0 %}{% set content = bos_token + content %}"
                "{% endif %}"
                "{{ content }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ 'assistant: ' }}"
                "{% endif %}"
            )
        }

        self.target_metric_train = "edit_success"
        self.target_metric_valid = self.target_metric_train
        self.target_metric_test = self.target_metric_train
        self.target_metric_transfer = self.target_metric_train
        self.has_transfer_split = False
        self.has_training_split = True
        
        self.data_path = data_path
        

    def get_train_data(
        self,
    ):
        with open(self.data_path, 'r') as file:
            data = json.load(file)[:50]
        
        total_ix = list(range(len(data)))
        # train and validation split makes sense on the editing item level
        return data, total_ix, total_ix

    def get_rewards(self, res):
        # Normalize each metric to ensure they're in [0, 1] range if not already
        # Assuming each metric is already normalized between 0 and 1
        
        # Apply weights to each component
        w_edit = 0.4  # Edit success weight
        w_locality = 0.3  # Locality weight
        w_portability = 0.3  # Portability weight
        
        # Calculate composite reward
        rewards = [
            w_edit * x['edit_success'] +
            w_locality * x['locality'] +
            w_portability * x['portability']
            for x in res.sample_details
        ]
        # Scale to [-1.0, 1.0] range
        rewards = [
            2 * (1 / (1 + math.exp(-reward / 0.5))) - 1 for reward in rewards
        ]
        return rewards


    def get_evaluator(
        self,
    ):
        with open(self.data_path, 'r') as file:
            samples = json.load(file)[:50]
        eval_task = KnowledgeEditingTask(
            samples,
            context_messages=[
                Message("system", KNOWLEDGE_ASSISTANT_PROMPT)
            ]
        )
        return [
            eval_task,
            eval_task
        ]
        
    def get_prompt(self, tokenizer, samples, ix, model_id):
        chat_template = self.model_to_template[model_id]
        context_msg = {"role": "system", "content": KNOWLEDGE_ASSISTANT_PROMPT}
        user_query = f"{samples[ix]['prompt']} {samples[ix]['target_new']}"
        user_msg = {"role": "user", "content": user_query}
        prompt = tokenizer.apply_chat_template(
            conversation=[context_msg, user_msg],
            chat_template=chat_template,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt

    def get_vllm_model(self, model_id) -> VLLMModel:
        """Load a vLLM model."""
        model = vllm.LLM(
            model_id,
            max_model_len=1024,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
            dtype="bfloat16",
            download_dir=get_download_dir(),
        )
        chat_template = self.model_to_template[model_id]
        # This may change with vLLM versions.
        m = model.llm_engine.model_executor.driver_worker.model_runner.model
        for _, param in m.named_parameters():
            param.requires_grad = False
        vllm_model = VLLMModel(
            model,
            sampling_params=vllm.SamplingParams(
                temperature=0,
                top_p=1,
                max_tokens=512,
                stop=["Instruction:", "Instruction", "Response:", "Response"],
                repetition_penalty=1.0,
            ),
            chat_template=chat_template,
        )
        return vllm_model
