import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from torch.optim import AdamW
from peft import get_peft_model, LoraConfig, TaskType
from tqdm.auto import tqdm
import hydra
from omegaconf import DictConfig

# Dataset preparation
class EditKnowDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        # Load the dataset from the JSON file
        with open(file_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # For training we use the "prompt" and "target_new" fields
        item = self.data[idx]
        text = item["prompt"] + " " + item["target_new"]
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].squeeze()       # Remove extra dimension
        attention_mask = inputs["attention_mask"].squeeze()
        return {"input_ids": input_ids, "attention_mask": attention_mask}

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Print the configuration for debugging
    print(cfg.pretty())

    # Load the tokenizer using the model_name from config
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as the padding token

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name,
        device_map=cfg.model.device_map,
        torch_dtype=getattr(torch, cfg.model.torch_dtype)
        # Optionally add: token=cfg.model.token, if using a HuggingFace token
    )

    # Configure LoRA via PEFT
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=cfg.lora.inference_mode,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout
    )
    model = get_peft_model(model, peft_config)

    # Create the dataset and DataLoader
    dataset = EditKnowDataset(cfg.dataset.file_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=cfg.training.learning_rate)
    num_epochs = cfg.training.num_epochs
    num_training_steps = num_epochs * len(dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    # Training loop with average loss per epoch
    progress_bar = tqdm(range(num_training_steps))
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss}")

    # Save the adapted model
    model.save_pretrained("lora_meta_llama_3.2_1B")

if __name__ == "__main__":
    main()
