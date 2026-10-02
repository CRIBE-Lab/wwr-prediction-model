"""Unified, reproducible training entry point for M1-M5."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from wwr_prediction.data import FacadeDataset, IGNORE_INDEX, discover_samples
from wwr_prediction.models import ResNet18UNet, build_model
from wwr_prediction.training import CombinedLoss, confusion_matrix, metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--stage", choices=["building", "opening"])
    parser.add_argument("--check", action="store_true", help="Run one forward/backward pass only")
    return parser.parse_args()


def resolve_task(config: dict, stage: str | None):
    if config["task"] == "single_stage":
        if stage:
            raise ValueError("--stage is only valid for M3/M4")
        return "single_stage", 3, config["class_weights"], -100
    if not stage:
        raise ValueError("M3/M4 require --stage building or --stage opening")
    if stage == "opening":
        return "opening", 2, config["stages"]["opening"]["class_weights"], IGNORE_INDEX
    task = "building_hard_negative" if config["task"] == "two_stage_hard_negative" else "building_binary"
    classes = 3 if task == "building_hard_negative" else 2
    return task, classes, config["stages"]["building"]["class_weights"], -100


def run_epoch(model, loader, loss_fn, device, num_classes, ignore_index, optimizer=None):
    model.train(optimizer is not None)
    total_loss = 0.0
    total_confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for images, masks, _ in tqdm(loader, leave=False):
        images, masks = images.to(device), masks.to(device)
        if optimizer:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(images)
            loss = loss_fn(logits, masks)
            if optimizer:
                loss.backward()
                optimizer.step()
        total_loss += float(loss) * images.size(0)
        total_confusion += confusion_matrix(logits.argmax(1).cpu(), masks.cpu(), num_classes, ignore_index)
    return {"loss": total_loss / len(loader.dataset), **metrics(total_confusion)}


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    task, num_classes, class_weights, ignore_index = resolve_task(config, args.stage)
    samples = discover_samples(args.data_dir, config.get("sample_limit"))
    if len(samples) < 2:
        raise ValueError("At least two image/JSON pairs are required")

    val_count = max(1, int(len(samples) * 0.2))
    train_subset, val_subset = random_split(
        samples, [len(samples) - val_count, val_count], generator=torch.Generator().manual_seed(seed)
    )
    train_samples = [samples[index] for index in train_subset.indices]
    val_samples = [samples[index] for index in val_subset.indices]
    train_loader = DataLoader(
        FacadeDataset(train_samples, task, config["image_size"], True),
        batch_size=config["batch_size"], shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        FacadeDataset(val_samples, task, config["image_size"], False),
        batch_size=config["batch_size"], shuffle=False, num_workers=0,
    )

    model = build_model(config["architecture"], num_classes, config.get("pretrained", False)).to(device)
    loss_fn = CombinedLoss(num_classes, class_weights, ignore_index).to(device)
    if isinstance(model, ResNet18UNet):
        optimizer = torch.optim.AdamW(
            [
                {"params": model.encoder_parameters(), "lr": config["encoder_learning_rate"]},
                {"params": model.decoder_parameters(), "lr": config["decoder_learning_rate"]},
            ], weight_decay=config["weight_decay"],
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    output_dir = args.output_dir / config["version"].lower() / (args.stage or "single_stage")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.check:
        images, masks, _ = next(iter(train_loader))
        loss = loss_fn(model(images.to(device)), masks.to(device))
        loss.backward()
        print({"status": "ok", "loss": float(loss), "samples": len(samples), "device": str(device)})
        return

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    best_loss = float("inf")
    best_epoch = 0
    wait = 0
    history = []
    for epoch in range(1, config["epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, loss_fn, device, num_classes, ignore_index, optimizer)
        val_metrics = run_epoch(model, val_loader, loss_fn, device, num_classes, ignore_index)
        scheduler.step(val_metrics["loss"])
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(json.dumps(history[-1], ensure_ascii=False))
        if val_metrics["loss"] < best_loss:
            best_loss, best_epoch, wait = val_metrics["loss"], epoch, 0
            torch.save(
                {"model_state_dict": model.state_dict(), "config": config, "task": task, "epoch": epoch},
                output_dir / "best_model.pth",
            )
        else:
            wait += 1
        (output_dir / "training_log.json").write_text(
            json.dumps({"best_epoch": best_epoch, "best_val_loss": best_loss, "history": history}, indent=2),
            encoding="utf-8",
        )
        if wait >= config["patience"]:
            break


if __name__ == "__main__":
    main()

