import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score


# ──────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────

def build_weighted_criterion(train_labels, num_classes, device):
    """
    Weighted cross-entropy to counteract class imbalance.
    w_c = N_total / (C * N_c)  — same formula as in the paper.
    """
    counts = np.bincount(train_labels, minlength=num_classes).astype(float)
    weights = len(train_labels) / (num_classes * counts)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    print("Class weights:", weights.cpu().numpy().round(2))
    return nn.CrossEntropyLoss(weight=weights)


# ──────────────────────────────────────────────
# One epoch
# ──────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device,
                    hsic_weight=0.1, epoch=0):
    model.train()
    run_loss = run_ce = run_hsic = correct = total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [train]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        logits, hsic_loss, _ = model(images, compute_uncertainty=False)
        ce_loss = criterion(logits, labels)
        loss = ce_loss + hsic_weight * hsic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = logits.argmax(1)
        total   += labels.size(0)
        correct += preds.eq(labels).sum().item()
        run_loss += loss.item(); run_ce += ce_loss.item(); run_hsic += hsic_loss.item()

        n = pbar.n + 1
        pbar.set_postfix(loss=f"{run_loss/n:.4f}",
                         ce=f"{run_ce/n:.4f}",
                         hsic=f"{run_hsic/n:.4f}",
                         acc=f"{100.*correct/total:.2f}%")

    return run_loss / len(loader), 100. * correct / total


# ──────────────────────────────────────────────
# Evaluation  — ALL metrics are macro-averaged
# ──────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, device, hsic_weight=0.1, desc="Val"):
    """
    Returns loss, accuracy, macro-F1, macro-recall, macro-precision.
    Macro averaging is used throughout to match Table 1 / Table 6 / Table 7.
    """
    model.eval()
    run_loss = correct = total = 0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc=desc):
        images, labels = images.to(device), labels.to(device)
        logits, hsic_loss, _ = model(images)
        loss = criterion(logits, labels) + hsic_weight * hsic_loss

        preds = logits.argmax(1)
        total   += labels.size(0)
        correct += preds.eq(labels).sum().item()
        run_loss += loss.item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc       = 100. * correct / total
    macro_f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_pre = precision_score(all_labels, all_preds, average='macro', zero_division=0)

    return run_loss / len(loader), acc, macro_f1, macro_rec, macro_pre


# ──────────────────────────────────────────────
# Full training loop
# ──────────────────────────────────────────────

def train_model(model, train_loader, val_loader, train_labels,
                num_classes=8, num_epochs=100, lr=1e-4,
                hsic_weight=0.1, device='cuda',
                save_path='best_model.pth'):

    model = model.to(device)
    criterion = build_weighted_criterion(train_labels, num_classes, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_f1 = 0.0
    history = {k: [] for k in ('train_loss', 'train_acc',
                                'val_loss', 'val_acc', 'val_f1')}

    for epoch in range(num_epochs):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, hsic_weight, epoch)

        val_loss, val_acc, val_f1, val_rec, val_pre = evaluate(
            model, val_loader, criterion, device, hsic_weight, desc="Val")

        scheduler.step()

        for k, v in zip(history, [tr_loss, tr_acc, val_loss, val_acc, val_f1]):
            history[k].append(v)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train  loss={tr_loss:.4f}  acc={tr_acc:.2f}%")
        print(f"  Val    loss={val_loss:.4f}  acc={val_acc:.2f}%")
        print(f"         Macro F1={val_f1:.4f}  Rec={val_rec:.4f}  Pre={val_pre:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print(f"  >> Best model saved  (Macro F1={best_f1:.4f})")

    return history
