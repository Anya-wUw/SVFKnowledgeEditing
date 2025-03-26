import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from ..models import GenerationRequest, Message, Model
from .base import Task, TaskResult


@dataclass
class EvalItem:
    prompt: str
    ground_truth: str

@dataclass
class PortabilityEval:
    reasoning: Optional[EvalItem]
    subject_aliasing: Optional[EvalItem]

@dataclass
class LocalityEval:
    relation_specificity: Optional[EvalItem]
    forgetfullness: Optional[EvalItem]

@dataclass
class KnowledgeEditingRequest:
    subject: str
    prompt: str
    target_new: str
    ground_truth: str
    portability: PortabilityEval
    locality: LocalityEval

def extract_answer_number(completion: str) -> Optional[float]:
    matches = re.findall(r"\d*\.?\d+", completion)
    if not matches:
        return None
    text = matches[-1]
    return float(text.replace(",", ""))

def mean(iterable: Iterable[float]) -> float:
    total, count = 0.0, 0
    for x in iterable:
        total += x
        count += 1
    return total / count

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

        requests = []
        for sample in samples:
            messages = list(self.context_messages)
            user_query = f"{sample['prompt']} {sample['target_new']}"
            messages.append(Message(role="user", content=user_query))
            requests.append(GenerationRequest(messages=messages))

        sample_details = []
        for sample, result in zip(samples, model.generate(requests)):
            output = result.generation
            prediction = extract_answer_number(result.generation)
            
            sample_details.append(
                dict(
                    prompt=sample['prompt'],
                    target_new=sample['target_new'],
                    prediction=prediction,
                    correct=prediction == sample['target_new']
                )
            )
            
        aggregate_metrics = {"acc": mean(sd["correct"] for sd in sample_details)}

        return TaskResult(
            aggregate_metrics=aggregate_metrics, sample_details=sample_details
        )
