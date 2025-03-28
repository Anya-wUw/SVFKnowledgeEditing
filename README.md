# Compositional-Knowledge-Editing-with-Transformer^2-and-SVF-Experts

This project explores a novel approach to knowledge editing in large language models (LLMs) using Transformer-Squared fine-tuning with Singular Value Factorization (SVF) experts. We aim to address the challenge of updating specific factual information in LLMs without full retraining, balancing edit precision with model stability.

Our method leverages SVF to modify only the singular values of weight matrices, achieving parameter-efficient updates while preserving the model's overall structure. We compare our approach against baseline methods such as LoRA, IKE, and prompt-based editing on the WikiData_counterfact dataset.

## Prerequistes

To reproduce our experiments with `meta-llama/Llama-3.2-1B` model you should have:
- 2 GPU devices of at least 24 GB VRAM (e.g. NVIDIA RTX 3090)
- CUDA >= 12.1 driver installed
- Docker installed

## Build and run
1. clone the repo ``git clone https://github.com/Anya-wUw/SVFKnowledgeEditing``
2. Move into the directory ``cd SVFKnowledgeEditing``
3. Build the container ``docker build -t svf_knowledge_editing`` .
4. [Optional] specify experiment config in the `config` folder.
5. Run the container ``docker run -it --gpus all svf_knowledge_editing``
6. Run the experiment ``cd svf_knowledge_editing && bash train_task_expert.sh``
7. Log in to wandb
8. Check our results in your wandb profile

## Results

Our experiments on the WikiData_counterfact dataset with Llama3.2_1B show:

| Method        | Edit Success | Locality | Portability |
|---------------|--------------|----------|-------------|
| SVF (ours)    | 76.3%        | 79.8%    | 71.2%       |
| LoRA          | 75.4%        | 82.1%    | 70.3%       |
| IKE           | 73.2%        | 80.5%    | 72.6%       |
| Prompt-based  | 67.8%        | 77.4%    | 66.5%       |

## Result Analysis

- SVF achieves the highest Edit Success (76.3%) while maintaining competitive Locality and Portability scores.
- LoRA shows strong Locality (82.1%) but may overfit on smaller datasets or sequential edits.
- IKE excels in Portability (72.6%) due to its dynamic, inference-time updates.
- The prompt-based approach consistently underperforms across all metrics.

Our SVF method demonstrates a promising balance between edit precision and model stability. However, challenges remain, including model size limitations, dataset constraints, and the risk of catastrophic forgetting with overlapping edits. Future work will focus on larger models, more comprehensive benchmarks, and advanced techniques to further isolate edited facts and minimize interference with existing knowledge.
