import os
import json
from PIL import Image
from torch.utils.data import Dataset,Subset,DataLoader
import torch
import random
from transformers import AutoProcessor, AutoTokenizer

class CaptionDataset(Dataset):
    def __init__(self, chat_json_path, image_root, processor, tokenizer, max_length=512,mode="train"):
        with open(chat_json_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

        self.image_root = image_root
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
    def __len__(self):
        return len(self.dataset)

    def _build_prompt(self, conversations):
        system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        question = conversations[0]["value"].replace(
                "<image>",
                "<|vision_start|><|image_pad|><|vision_end|>"
            )
        caption = conversations[1]["value"].strip()
        prompt = f"{system_prompt}<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        full_text = f"{prompt}{caption}<|im_end|>"

        return prompt, full_text, caption

    def __getitem__(self, idx):
        item = self.dataset[idx]

        image_name = item["image"]
        image_path = os.path.join(self.image_root, image_name)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        conversations = item["conversations"]

        prompt, full_text,caption = self._build_prompt(conversations)

        image_inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].squeeze(0)

        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )

        prompt_enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        if self.mode == "train":
            input_ids = full_enc["input_ids"].squeeze(0)
            attention_mask = full_enc["attention_mask"].squeeze(0)
        else:
            input_ids = prompt_enc["input_ids"].squeeze(0)
            attention_mask = prompt_enc["attention_mask"].squeeze(0)
        prompt_len = prompt_enc["input_ids"].size(1)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_len": prompt_len,
            "reference":caption,
            "image_path":image_path
        }

def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8

    n = len(dataset)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)

class MultiModalCollator:
    def __init__(self, tokenizer, max_length=None, padding_side: str = "right"):
        """
        padding_side:
            - "right"：训练/计算 loss 时使用，labels 用右 padding 对齐。
            - "left" ：batch 生成时必须使用，否则 causal LM 的新 token 会被追加到 pad 之后，
                       导致短 prompt 的样本产出乱码（常见现象：prompt tail 泄漏，比如输出里冒出 "assistant"）。
        """
        assert padding_side in ("right", "left")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding_side = padding_side

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch):
        pixel_values = torch.stack([x["pixel_values"] for x in batch], dim=0)
        references = [x["reference"] for x in batch]

        input_ids_list = [x["input_ids"] for x in batch]
        attention_mask_list = [x["attention_mask"] for x in batch]
        prompt_lens = [x["prompt_len"] for x in batch]

        max_seq_len = max(x.size(0) for x in input_ids_list)
        if self.max_length is not None:
            max_seq_len = min(max_seq_len, self.max_length)

        pad_token_id = self.tokenizer.pad_token_id

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for input_ids, attention_mask, prompt_len in zip(input_ids_list, attention_mask_list, prompt_lens):
            input_ids = input_ids[:max_seq_len]
            attention_mask = attention_mask[:max_seq_len]
            prompt_len = min(prompt_len, max_seq_len)

            seq_len = input_ids.size(0)
            pad_len = max_seq_len - seq_len

            pad_ids = torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype)
            pad_mask = torch.zeros(pad_len, dtype=attention_mask.dtype)
            pad_labels = torch.full((pad_len,), -100, dtype=input_ids.dtype)

            labels_core = input_ids.clone()
            labels_core[:prompt_len] = -100

            if self.padding_side == "right":
                padded_ids = torch.cat([input_ids, pad_ids])
                padded_mask = torch.cat([attention_mask, pad_mask])
                labels = torch.cat([labels_core, pad_labels]) if pad_len > 0 else labels_core
            else:  # left padding —— 生成专用
                padded_ids = torch.cat([pad_ids, input_ids])
                padded_mask = torch.cat([pad_mask, attention_mask])
                labels = torch.cat([pad_labels, labels_core]) if pad_len > 0 else labels_core

            padded_input_ids.append(padded_ids)
            padded_attention_mask.append(padded_mask)
            padded_labels.append(labels)

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.stack(padded_input_ids, dim=0),
            "attention_mask": torch.stack(padded_attention_mask, dim=0),
            "labels": torch.stack(padded_labels, dim=0),
            "references":references
        }

def build_dataloders(
    vision_model_name: str,
    qwen_model_name: str,
    chat_json_path: str,
    image_root: str,
    batch_size: int=8,
    num_workers: int = 4,
    max_length: int = 64,
    train_ratio: float=0.8,
    val_ratio: float=0.1,
    test_ratio:float=0.1
):
    processor = AutoProcessor.from_pretrained(vision_model_name)
    tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = CaptionDataset(
        chat_json_path=chat_json_path,
        image_root=image_root,
        processor=processor,
        tokenizer=tokenizer,
        max_length=max_length,
        mode="train"
    )

    eval_dataset = CaptionDataset(
        chat_json_path=chat_json_path,
        image_root=image_root,
        processor=processor,
        tokenizer=tokenizer,
        max_length=max_length,
        mode="eval"
    )
    n = len(train_dataset)
    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    train_ds = Subset(train_dataset, train_idx)
    val_ds = Subset(train_dataset, val_idx)
    test_ds = Subset(eval_dataset, test_idx)

    # 训练/算 loss 用右 padding；生成用左 padding（否则 batch 生成会产出乱码）
    train_collator = MultiModalCollator(tokenizer, max_length=max_length, padding_side="right")
    eval_collator = MultiModalCollator(tokenizer, max_length=max_length, padding_side="left")

    pin_memory = torch.cuda.is_available()
    persistent = num_workers > 0

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=train_collator,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=train_collator,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=eval_collator,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent,
    )
    return train_loader,val_loader,test_loader
    
if __name__=="__main__":
    processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = CaptionDataset(
        chat_json_path="../../llava_cc3m_raw/chat.json",
        image_root="../../llava_cc3m_raw/images",
        processor=processor,
        tokenizer=tokenizer,
        max_length=512,
    )

    train_ds, val_ds, test_ds = split_dataset(dataset, 0.8, 0.1, 0.1, seed=42)

    collator = MultiModalCollator(tokenizer, max_length=512)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collator, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collator, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collator, num_workers=4)

    for i in range(10):
        sample = test_ds[i]
        print(f"\n=== sample {i} ===")
        reference = sample["reference"]
        image_path = sample["image_path"]
        print(f"reference: {reference}")
        print(f"image_path: {image_path}")
