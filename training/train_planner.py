"""Train a planner action classifier on Hyperscape behavior tick data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from training.models.planner_model import (
    ACTION_LABELS,
    PlannerActionClassifier,
    load_dataset,
)


def train(
    dataset_path: Path,
    output_path: Path,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 0.001,
) -> dict:
    features, labels = load_dataset(dataset_path)
    if len(features) == 0:
        print("No valid training data found")
        return {"error": "no_data"}

    print(f"Dataset: {len(features)} rows, {features.shape[1]} features")
    label_counts = Counter(labels.tolist())
    for idx, count in sorted(label_counts.items()):
        print(f"  {ACTION_LABELS[idx]}: {count}")

    # Train/val split (80/20)
    n = len(features)
    indices = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = indices[:split], indices[split:]

    X_train = torch.from_numpy(features[train_idx])
    y_train = torch.from_numpy(labels[train_idx]).long()
    X_val = torch.from_numpy(features[val_idx])
    y_val = torch.from_numpy(labels[val_idx]).long()

    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Class weights for imbalanced data
    class_counts = np.bincount(labels, minlength=len(ACTION_LABELS)).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(ACTION_LABELS)
    weights_tensor = torch.from_numpy(class_weights)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = PlannerActionClassifier(input_dim=features.shape[1])
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == y_val).float().mean().item()
            train_logits = model(X_train)
            train_preds = train_logits.argmax(dim=1)
            train_acc = (train_preds == y_train).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: loss={total_loss/len(train_loader):.4f} "
                  f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} "
                  f"best_val_acc={best_val_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val)
        val_preds = val_logits.argmax(dim=1)

    # Per-class accuracy
    print("\nPer-class results:")
    for idx, label in enumerate(ACTION_LABELS):
        mask = y_val == idx
        if mask.sum() == 0:
            continue
        class_acc = (val_preds[mask] == idx).float().mean().item()
        print(f"  {label}: {class_acc:.3f} ({mask.sum().item()} val samples)")

    # Save checkpoint
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "planner_classifier_v1",
        "model_state_dict": model.state_dict(),
        "input_dim": features.shape[1],
        "num_actions": len(ACTION_LABELS),
        "action_labels": ACTION_LABELS,
        "best_val_acc": best_val_acc,
        "train_size": len(X_train),
        "val_size": len(X_val),
    }
    torch.save(checkpoint, output_path)
    print(f"\nSaved checkpoint to {output_path}")
    print(f"Best validation accuracy: {best_val_acc:.3f}")

    return {
        "best_val_acc": best_val_acc,
        "train_size": len(X_train),
        "val_size": len(X_val),
    }


def main():
    parser = argparse.ArgumentParser(description="Train planner action classifier")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, default="end_to_end_outputs/policies/planner_classifier.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    train(
        dataset_path=Path(args.dataset),
        output_path=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
