from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, List, Dict

from ..models import GenerationRequest, Message, Model
from .base import Task, TaskResult


@dataclass
class EvalItem:
    prompt: str
    ground_truth: List[str]

@dataclass
class PortabilityEval:
    reasoning: Optional[List[EvalItem]]
    subject_aliasing: Optional[List[EvalItem]]

@dataclass
class LocalityEval:
    relation_specificity: Optional[List[EvalItem]]
    forgetfullness: Optional[List[EvalItem]]

@dataclass
class KnowledgeEditingRequest:
    subject: str
    prompt: str
    target_new: str
    ground_truth: str
    portability: PortabilityEval
    locality: LocalityEval

class KnowledgeEditingTask(Task):
    def __init__(
        self,
        samples: Sequence[KnowledgeEditingRequest],
        context_messages: Sequence[Message] = (),
    ):
        self.samples = list(samples)
        self.context_messages = context_messages

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def evaluate(
        self,
        model: Model,
        sample_ids: Optional[Sequence[int]] = None,
    ) -> TaskResult:
        if sample_ids is None:
            sample_ids = range(len(self.samples))
        samples = [self.samples[sample_id] for sample_id in sample_ids]

        edit_success, edit_success_sample_details = self.compute_edit_success(model, samples)
        locality, locality_sample_details = self.compute_locality(model, samples)
        portability, portability_sample_details = self.compute_portability(model, samples)

        sample_details = [
            dict(
                prompt=sample['prompt'],
                target_new=sample['target_new'],
                prediction=edit_success_details['prediction'],
                edit_success=edit_success_details['edit_success'],
                locality=locality_details['locality'],
                portability=portabiliy_details['portability']
            )
            for sample, edit_success_details, locality_details, portabiliy_details
            in zip(samples, edit_success_sample_details, locality_sample_details, portability_sample_details)
        ]
        
        aggregate_metrics = {
            "edit_success": edit_success,
            "locality": locality,
            "portability": portability
        }

        return TaskResult(
            aggregate_metrics=aggregate_metrics, sample_details=sample_details
        )
    
    def compute_edit_success(
        self,
        model: Model,
        samples: Sequence[KnowledgeEditingRequest]
    ) -> List[Dict]:
        requests = []
        correct_count = 0
        
        sample_details = []
        
        for sample in samples:
            messages = list(self.context_messages)
            user_query = sample['prompt']
            messages.append(Message(role="user", content=user_query))
            requests.append(GenerationRequest(messages=messages))

        for sample, result in zip(samples, model.generate(requests)):
            prediction = result.generation
            edit_success = 0
            if sample['target_new'] in prediction.split():
                correct_count += 1
                edit_success = 1
            sample_details.append({'edit_success': edit_success, 'prediction': prediction})
        
        total_count = len(requests)
        return correct_count / total_count if total_count > 0 else 0.0, sample_details
    
    def compute_locality(
        self,
        model: Model,
        samples: Sequence[KnowledgeEditingRequest]
    ) -> float:
        request_samples: List[KnowledgeEditingRequest] = []
        requests: List[GenerationRequest] = []
        ground_truth_items: List[List[str]] = []
        
        for sample in samples:
            # Process relation specificity prompts
            eval_items: List[EvalItem] = []
            if sample['locality'].get('Relation_Specificity'):
                eval_items += sample['locality'].get('Relation_Specificity')
            if sample['locality'].get('Forgetfulness'):
                eval_items += sample['locality'].get('Forgetfulness')
            
            for eval_item in eval_items:
                messages = list(self.context_messages)
                messages.append(Message(role="user", content=eval_item['prompt']))
                requests.append(GenerationRequest(messages=messages))
                ground_truth_items.append(eval_item['ground_truth'])
                request_samples.append(sample)
        
        metric_value, sample_correctness = self._compute_metric(requests, ground_truth_items, request_samples, model)
        sample_details = []
        
        for sample in samples:
            locality = sample_correctness.get(sample['subject'])
            locality = locality if locality else 0.0
            sample_details.append({'locality': locality})
        
        return metric_value, sample_details

    def compute_portability(
        self,
        model: Model,
        samples: Sequence[KnowledgeEditingRequest]
    ) -> float:
        request_samples: List[KnowledgeEditingRequest] = []
        requests: List[GenerationRequest] = []
        ground_truth_items: List[List[str]] = []
        
        for sample in samples:
            # Process relation specificity prompts
            eval_items: List[EvalItem] = []
            if sample['portability'].get('Reasoning'):
                eval_items += sample['portability'].get('Reasoning')
            if sample['portability'].get('Subject_Aliasing'):
                eval_items += sample['portability'].get('Subject_Aliasing')
            
            for eval_item in eval_items:
                messages = list(self.context_messages)
                messages.append(Message(role="user", content=eval_item['prompt']))
                requests.append(GenerationRequest(messages=messages))
                ground_truth_items.append(eval_item['ground_truth'])
                request_samples.append(sample)
        
        metric_value, sample_correctness = self._compute_metric(requests, ground_truth_items, request_samples, model)
        sample_details = []
        
        for sample in samples:
            portability = sample_correctness.get(sample['subject'])
            portability = portability if portability else 0.0
            sample_details.append({'portability': portability})
        
        return metric_value, sample_details

    def _compute_metric(
        self,
        requests: List[GenerationRequest],
        ground_truth_items: List[List[str]],
        request_samples: Sequence[KnowledgeEditingRequest],
        model: Model) -> float:
        correct_count = 0
        sample_correctness: Dict[str, bool] = {}
        for sample in request_samples:
            sample_correctness[sample['subject']] = False

        for ground_truth, result, sample in zip(ground_truth_items, model.generate(requests), request_samples):
            prediction = result.generation
            if any(gt for gt in ground_truth if gt in prediction.split()):
                correct_count += 1
                sample_correctness[sample['subject']] = True
        
        total_count = len(requests)
        return correct_count / total_count if total_count > 0 else 0.0, sample_correctness
