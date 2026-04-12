import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.multimodal_model import create_multimodal_model
from data.caption_dataset import build_dataloders
from eval import evaluate_generation, print_caption_metrics
from utils.checkpoint import ensure_dir, save_checkpoint, load_checkpoint

from tqdm import tqdm
import wandb
import os

import argparse
import yaml

@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
) -> float:
    """
    执行一次完整验证

    Args:
        model: 模型
        dataloader: 验证集 dataloader
        device: 设备
        epoch: 当前 epoch

    Returns:
        平均验证 loss
    """
    model.eval()

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(
        dataloader,
        desc=f"Val   Epoch {epoch + 1:02d}",
        leave=False,
        dynamic_ncols=True,
    )

    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs["loss"]

        total_loss += loss.item()
        num_batches += 1

        avg_loss = total_loss / num_batches
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_loss:.4f}")

    return total_loss / max(num_batches, 1)

def train(model: nn.Module, train_loader : DataLoader, val_loader : DataLoader, 
    optimizer: torch.optim.Optimizer, scheduler, device: torch.device,
    start_epoch:int, num_epochs: int, config: dict,best_val_loss:float):

    save_dir = config.get("save_dir", "./outputs")
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    best_dir = os.path.join(save_dir, "best_model")
    final_dir = os.path.join(save_dir, "final_model")

    ensure_dir(save_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(best_dir)
    ensure_dir(final_dir)
    
    use_wandb = config.get("use_wandb", True)
    # 初始化 wandb
    if use_wandb:
        wandb.init(
            project=config.get("project_name", "multimodal-caption"),
            name=config.get("run_name", None),
            config=config,
            mode=config.get("wandb_mode", "online"),
        )
        # 记录一个便于观察的指标轴
        wandb.define_metric("train/global_step")
        wandb.define_metric("train/*", step_metric="train/global_step")
        wandb.define_metric("eval/global_step")
        wandb.define_metric("eval/*", step_metric="eval/global_step")

        # 监控梯度/参数，可选，频率不宜太高
        if config.get("wandb_watch", False):
            wandb.watch(model, log="all", log_freq=config.get("logging_steps", 10))

    for epoch in range(start_epoch, num_epochs):

        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            train_loader,
            desc=f"Train Epoch {epoch + 1:02d}",
            leave=True,
            dynamic_ncols=True,
        )
        
        for step, batch in enumerate(pbar):
            global_step = epoch * len(train_loader) + step

            # 移动数据到设备
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            # print(pixel_values.shape)
            # print(input_ids.shape)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs["loss"]
            # 反向传播
            loss.backward()

            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            # 记录损失
            total_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            num_batches += 1
            avg_loss = total_loss / num_batches

            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_loss:.4f}", lr=f"{current_lr:.2e}")
            # 日志记录
            if global_step % config['logging_steps'] == 0:
                log_dict = {
                    "train/global_step": global_step,
                    "train/epoch": epoch + 1,
                    "train/loss": loss.item(),
                    "train/avg_loss": avg_loss,
                    "train/lr": current_lr,
                    "train/grad_norm": float(grad_norm) if grad_norm is not None else 0.0,
                }

                if use_wandb:
                    wandb.log(log_dict)
            # 验证和保存检查点
            if global_step % config['eval_steps'] == 0 and global_step > 0:
                val_loss = validate(model, val_loader, device, epoch)
                print(f"[Eval @ step {global_step}] val_loss={val_loss:.4f}")
                if use_wandb:
                    wandb.log({
                        "eval/global_step": global_step,
                        "eval/epoch": epoch + 1,
                        "eval/loss": val_loss,
                    })

                # 保存最佳模型
                if val_loss < best_val_loss-0.0001:
                    best_val_loss = val_loss
                    best_ckpt_path = os.path.join(best_dir, "best_checkpoint.pt")
                
                    save_checkpoint(
                        save_path=best_ckpt_path,
                        model=model,
                        optimizer=None,
                        scheduler=None,
                        epoch=epoch,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                        config=config,
                        extra={"best_from": "step_eval"},
                    )

                    print(f"[Best Model Saved] step={global_step}, val_loss={val_loss:.4f}")

                    if use_wandb:
                        wandb.log({
                            "eval/global_step": global_step,
                            "eval/best_val_loss": best_val_loss,
                        })
                    
                model.train()
            save_steps = config.get("save_steps", None)
            if save_steps is not None and save_steps > 0:
                if global_step % save_steps == 0 and global_step > 0:
                    ckpt_path = os.path.join(ckpt_dir, f"checkpoint-step-{global_step}.pt")
                    save_checkpoint(
                        save_path=ckpt_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                        config=config,
                    )
                    print(f"[Checkpoint Saved] {ckpt_path}")
        
        epoch_avg_loss = total_loss / max(num_batches, 1)
        end_of_epoch_step = (epoch + 1) * len(train_loader) - 1
        
        # 每个 epoch 结束后做一次验证
        val_loss = validate(model, val_loader, device, epoch)
        print(
            f"[Epoch {epoch + 1}/{num_epochs}] "
            f"train_avg_loss={epoch_avg_loss:.4f} | val_loss={val_loss:.4f}"
        )
        if use_wandb:
            wandb.log({
                "eval/global_step": end_of_epoch_step,
                "eval/epoch": epoch + 1,
                "eval/epoch_train_avg_loss": epoch_avg_loss,
                "eval/loss": val_loss,
            })
        # 保存最佳模型
        if val_loss < best_val_loss-0.0001:
            best_val_loss = val_loss
            
            best_ckpt_path = os.path.join(best_dir, "best_checkpoint.pt")
            
            save_checkpoint(
                save_path=best_ckpt_path,
                model=model,
                optimizer=None,
                scheduler=None,
                epoch=epoch,
                global_step=end_of_epoch_step,
                best_val_loss=best_val_loss,
                config=config,
                extra={"best_from": "epoch_eval"},
            )
            
            print(f"[Best Model Saved] epoch={epoch + 1}, val_loss={val_loss:.4f}")

            if use_wandb:
                wandb.log({
                    "eval/global_step": end_of_epoch_step,
                    "eval/best_val_loss": best_val_loss,
                })
        # 保存检查点
        # 每个 epoch 结束保存一个 checkpoint
        epoch_ckpt_path = os.path.join(ckpt_dir, f"checkpoint-epoch-{epoch + 1}.pt")
        save_checkpoint(
            save_path=epoch_ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=end_of_epoch_step,
            best_val_loss=best_val_loss,
            config=config,
        )
        print(f"[Epoch Checkpoint Saved] {epoch_ckpt_path}")
    # 保存最终模型
    final_ckpt_path = os.path.join(final_dir, "final_checkpoint.pt")
    
    final_global_step = num_epochs * len(train_loader) - 1

    save_checkpoint(
        save_path=final_ckpt_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=num_epochs - 1,
        global_step=final_global_step,
        best_val_loss=best_val_loss,
        config=config,
        extra={"is_final": True},
    )
 
    print(f"[Final Checkpoint Saved] {final_ckpt_path}")

    if use_wandb:
        wandb.summary["best_val_loss"] = best_val_loss
        wandb.summary["final_epoch"] = num_epochs
        
    return best_val_loss

def main():
    parser = argparse.ArgumentParser(description="Stage 1 Training: Alignment")
    parser.add_argument("--config", type=str, default="./config/training_stage2.yaml",
                       help="Path to config file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")

    args = parser.parse_args()
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建模型
    model = create_multimodal_model(config)
    model.to(device)

    train_loader,val_loader,test_loader = build_dataloders(
        vision_model_name=config["dataset"]["vision_model_name"],
        qwen_model_name=config["dataset"]["qwen_model_name"],
        chat_json_path=config["dataset"]["chat_json_path"],
        image_root=config["dataset"]["image_root"],
        batch_size=config["dataset"]["batch_size"],
        num_workers=config["dataset"]["num_workers"],
        max_length=config["dataset"]["max_length"],
        train_ratio=config["dataset"]["train_ratio"],
        val_ratio=config["dataset"]["val_ratio"],
        test_ratio=config["dataset"]["test_ratio"]
    )
    # 创建优化器
    optimizer_config = config['optimizer']
    if optimizer_config['name'] == 'adamw':
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config['weight_decay'],
            betas=tuple(optimizer_config['betas'])
        )
    num_epochs = config['epochs']
    # 创建学习率调度器
    scheduler_config = config['scheduler']
    if scheduler_config['name'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs * len(train_loader),
            eta_min=0
        )
    else:
        scheduler = None

    # 恢复检查点 (如果指定)
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(
            f"Resumed from checkpoint: {args.resume} | "
            f"start_epoch={start_epoch} | best_val_loss={best_val_loss:.6f}"
        )
    best_val_loss = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        start_epoch=start_epoch,
        num_epochs=num_epochs,
        config=config,
        best_val_loss=best_val_loss,
    )

    print(f"Training finished. Best val loss: {best_val_loss:.6f}")

    # ===== 测试集 caption 评估 =====
    tokenizer = AutoTokenizer.from_pretrained(config["dataset"]["qwen_model_name"], use_fast=True)

    test_metrics = evaluate_generation(
        model=model,
        dataloader=test_loader,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=config.get("generation", {}).get("max_new_tokens", 64),
        num_beams=config.get("generation", {}).get("num_beams", 1),
        do_sample=config.get("generation", {}).get("do_sample", False),
        save_json_path=os.path.join(config.get("save_dir", "./outputs"), "test_predictions.json"),
        log_to_wandb=config.get("use_wandb", True),
        prefix="test",
    )
    print_caption_metrics(test_metrics, prefix="Test")
    use_wandb = config.get("use_wandb", True)
    if use_wandb:
        wandb.finish()
if __name__ == "__main__":
    main()