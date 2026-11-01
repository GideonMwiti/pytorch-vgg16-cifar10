"""
VGG-16 Deep Convolutional Classifier for CIFAR-10
Part of the '100 PyTorch Projects in 10 Days' Challenge (Day 3 - Project 23)
Author: Gideon Mwiti

Demonstrates:
- 16-Layer VGG Architecture (13 Conv + 3 Dense) adapted for 32x32 inputs
- Stacked 3x3 convolutions with effective receptive field factorization
- Batch Normalization & Dropout regularization for deep optimization
- Comprehensive diagnostic visualization with confusion matrix and prediction gallery
"""

import sys
import types
import io
import os
import random
import time
from typing import Tuple, List, Dict, Any

# Windows Python 3.11+ bz2/lzma compatibility patch
sys.modules['bz2'] = types.SimpleNamespace(open=io.open, BZ2File=io.BytesIO)
sys.modules['lzma'] = types.SimpleNamespace(
    open=io.open,
    LZMAFile=io.BytesIO,
    FORMAT_ALONE=1,
    FORMAT_XZ=2,
    CHECK_NONE=0,
    CHECK_CRC32=1,
    CHECK_CRC64=4,
    CHECK_SHA256=10
)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# -----------------------------------------------------------------------------
# 1. Reproducibility & Device Configuration
# -----------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Execution Device: {DEVICE}", flush=True)

CIFAR10_CLASSES = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


# -----------------------------------------------------------------------------
# 2. VGG-16 Architecture Adapted for CIFAR-10 (32x32 Inputs)
# -----------------------------------------------------------------------------
class VGG16Small(nn.Module):
    """
    16-layer VGG architecture (Simonyan & Zisserman, 2014) adapted for 32x32 inputs.
    Contains exactly 13 Conv layers + 3 Dense layers organized into 5 stages:
    - Stage 1: Conv3-32, Conv3-32, MaxPool(2x2) -> 16x16
    - Stage 2: Conv3-64, Conv3-64, MaxPool(2x2) -> 8x8
    - Stage 3: Conv3-128, Conv3-128, Conv3-128, MaxPool(2x2) -> 4x4
    - Stage 4: Conv3-256, Conv3-256, Conv3-256, MaxPool(2x2) -> 2x2
    - Stage 5: Conv3-256, Conv3-256, Conv3-256, MaxPool(2x2) -> 1x1
    - Classifier: Dense(256, 512) -> Dense(512, 512) -> Dense(512, 10)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        # Block 1 (32x32 -> 16x16)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 2 (16x16 -> 8x8)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 3 (8x8 -> 4x4)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 4 (4x4 -> 2x2)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 5 (2x2 -> 1x1)
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 3-Layer Dense Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(256 * 1 * 1, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# -----------------------------------------------------------------------------
# 3. Model Training & Evaluation Routines
# -----------------------------------------------------------------------------
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer
) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)
    return total_loss / total, (correct / total) * 100.0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, List[torch.Tensor]]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    y_true_list, y_pred_list, y_prob_list = [], [], []
    raw_images = []

    for images, targets in loader:
        images_dev = images.to(DEVICE)
        outputs = model(images_dev)
        loss = criterion(outputs, targets.to(DEVICE))

        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        correct += (preds == targets.numpy()).sum()
        total += targets.size(0)

        y_true_list.extend(targets.numpy())
        y_pred_list.extend(preds)
        y_prob_list.extend(probs)
        raw_images.extend([img for img in images])

    return (
        total_loss / total,
        (correct / total) * 100.0,
        np.array(y_true_list),
        np.array(y_pred_list),
        np.array(y_prob_list),
        raw_images
    )


# -----------------------------------------------------------------------------
# 4. Publication-Grade Visualization Dashboard
# -----------------------------------------------------------------------------
def plot_vgg16_dashboard(
    history: Dict[str, List[float]],
    cm_norm: np.ndarray,
    report: Dict[str, Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    raw_images: List[torch.Tensor],
    output_path: str = "vgg16_cifar10_results.png"
) -> None:
    print("[*] Generating publication-grade VGG-16 diagnostic dashboard...", flush=True)
    fig = plt.figure(figsize=(20, 13), dpi=300)
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], wspace=0.25, hspace=0.35)

    epochs = range(1, len(history["train_loss"]) + 1)

    # 1. Loss Dynamics
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#2980b9", linewidth=2.5, marker="o")
    ax1.plot(epochs, history["test_loss"], label="Val Loss", color="#e74c3c", linewidth=2.5, marker="s", linestyle="--")
    ax1.set_title("Cross-Entropy Loss Dynamics (16 Layers)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Loss", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=9.5, loc="upper right")

    # 2. Accuracy Trajectory
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history["train_acc"], label="Train Accuracy", color="#27ae60", linewidth=2.5, marker="o")
    ax2.plot(epochs, history["test_acc"], label="Val Accuracy", color="#8e44ad", linewidth=2.5, marker="^", linestyle="--")
    final_test_acc = history["test_acc"][-1]
    ax2.set_title(f"Accuracy Trajectory (Final Val: {final_test_acc:.2f}%)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=9.5, loc="lower right")

    # 3. Normalized Confusion Matrix Heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    im = ax3.imshow(cm_norm, cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("Recall Rate", fontsize=9.5, fontweight="bold")

    ax3.set_xticks(range(10))
    ax3.set_yticks(range(10))
    ax3.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right", fontsize=8.5, fontweight="bold")
    ax3.set_yticklabels(CIFAR10_CLASSES, fontsize=8.5, fontweight="bold")
    ax3.set_xlabel("Predicted Label", fontsize=10.5, fontweight="bold")
    ax3.set_ylabel("True Label", fontsize=10.5, fontweight="bold")
    ax3.set_title("VGG-16 Normalized Confusion Matrix", fontsize=12, fontweight="bold", pad=10)

    for i in range(10):
        for j in range(10):
            val = cm_norm[i, j]
            color = "white" if val > 0.4 else "black"
            ax3.text(j, i, f"{val*100:.0f}%", ha="center", va="center", color=color,
                     fontsize=7.5, fontweight="bold" if i == j else "normal")

    # 4. Receptive Field & Kernel Factorization Theory Diagram
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis("off")
    diagram_text = (
        "VGG Architectural Principle:\n"
        "Stacking Small 3x3 Filters vs. Larger Kernels\n\n"
        "• Two 3x3 Conv Stacks:\n"
        "  - Effective Receptive Field: 5x5\n"
        "  - Number of Non-linearities: 2 ReLUs\n"
        "  - Parameters: 2 × (3² × C²) = 18 C²\n"
        "  - 5x5 Kernel: 1 × (5² × C²) = 25 C²\n"
        "  - Savings: 28% fewer parameters!\n\n"
        "• Three 3x3 Conv Stacks:\n"
        "  - Effective Receptive Field: 7x7\n"
        "  - Number of Non-linearities: 3 ReLUs\n"
        "  - Parameters: 3 × (3² × C²) = 27 C²\n"
        "  - 7x7 Kernel: 1 × (7² × C²) = 49 C²\n"
        "  - Savings: 45% fewer parameters!\n\n"
        "Conclusion: Deeper networks with small filters learn\n"
        "more discriminative, non-linear feature representations\n"
        "with substantially reduced parameter budgets."
    )
    ax4.text(0.05, 0.95, diagram_text, transform=ax4.transAxes, fontsize=9.5,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#2980b9", alpha=0.9))
    ax4.set_title("Receptive Field Factorization Theory", fontsize=12, fontweight="bold", pad=12)

    # 5. Per-Class Precision / Recall / F1 Bar Chart
    ax5 = fig.add_subplot(gs[1, 1])
    precisions = [report[cls]["precision"] * 100 for cls in CIFAR10_CLASSES]
    recalls = [report[cls]["recall"] * 100 for cls in CIFAR10_CLASSES]
    f1_scores = [report[cls]["f1-score"] * 100 for cls in CIFAR10_CLASSES]

    x_idx = np.arange(10)
    width = 0.26
    ax5.bar(x_idx - width, precisions, width, label="Precision", color="#3498db", alpha=0.9)
    ax5.bar(x_idx, recalls, width, label="Recall", color="#2ecc71", alpha=0.9)
    ax5.bar(x_idx + width, f1_scores, width, label="F1-Score", color="#e67e22", alpha=0.9)

    ax5.set_xticks(x_idx)
    ax5.set_xticklabels([c[:4] for c in CIFAR10_CLASSES], fontsize=9, fontweight="bold")
    ax5.set_ylim(0, 100)
    ax5.set_xlabel("Class", fontsize=10.5, fontweight="bold")
    ax5.set_ylabel("Score (%)", fontsize=10.5, fontweight="bold")
    ax5.set_title("Class-Wise Performance Decomposition", fontsize=12, fontweight="bold", pad=10)
    ax5.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax5.legend(loc="lower right", fontsize=8.5)

    # 6. Qualitative Prediction Gallery
    sub_gs_pred = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[1, 2], wspace=0.25, hspace=0.35)
    inv_mean = [-m / s for m, s in zip(CIFAR_MEAN, CIFAR_STD)]
    inv_std = [1.0 / s for s in CIFAR_STD]
    denorm = T.Compose([
        T.Normalize(mean=[0.0, 0.0, 0.0], std=inv_std),
        T.Normalize(mean=inv_mean, std=[1.0, 1.0, 1.0]),
    ])

    for idx in range(8):
        sub_ax = fig.add_subplot(sub_gs_pred[idx // 4, idx % 4])
        img_tensor = raw_images[idx]
        disp_img = torch.clamp(denorm(img_tensor), 0.0, 1.0).permute(1, 2, 0).numpy()

        sub_ax.imshow(disp_img)
        t_cls = CIFAR10_CLASSES[y_true[idx]]
        p_cls = CIFAR10_CLASSES[y_pred[idx]]
        conf = y_prob[idx, y_pred[idx]] * 100.0

        is_correct = (y_true[idx] == y_pred[idx])
        color = "#27ae60" if is_correct else "#c0392b"
        sub_ax.set_title(f"T:{t_cls[:4]}\nP:{p_cls[:4]} ({conf:.0f}%)",
                         fontsize=7.5, fontweight="bold", color=color)
        sub_ax.axis("off")
        for spine in sub_ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.5)

    ax6_title = fig.add_subplot(gs[1, 2])
    ax6_title.axis("off")
    ax6_title.set_title("Qualitative CIFAR-10 Test Predictions", fontsize=12, fontweight="bold", pad=12)

    plt.suptitle("VGG-16 Deep Convolutional Architecture (16 Layers) on CIFAR-10",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[+] Diagnostic dashboard saved successfully to {output_path}", flush=True)


# -----------------------------------------------------------------------------
# 5. Main Routine
# -----------------------------------------------------------------------------
def main() -> None:
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    print("================================================================================")
    print("      VGG-16 Deep Convolutional Classifier (16 Layers) on CIFAR-10              ")
    print("================================================================================")

    # 1. Dataset Preparation with Data Augmentation
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
    ])

    train_dataset = CIFAR10(root=data_dir, train=True, download=False, transform=train_transform)
    test_dataset = CIFAR10(root=data_dir, train=False, download=False, transform=test_transform)

    # Use structured subsets for rapid CPU benchmarking
    train_subset = Subset(train_dataset, list(range(8000)))
    test_subset = Subset(test_dataset, list(range(1500)))

    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=128, shuffle=False, num_workers=0)
    print(f"[*] Train Subset: {len(train_subset):,} samples | Test Subset: {len(test_subset):,} samples.", flush=True)

    # 2. Model Instantiation & Parameter Verification
    set_seed(42)
    model = VGG16Small(num_classes=10).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] VGG-16 Instantiated: Total Trainable Parameters = {total_params:,}", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    num_epochs = 4

    # 3. Training Loop
    print(f"\n[*] Training VGG-16 for {num_epochs} Epochs on CIFAR-10...", flush=True)
    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc, _, _, _, _ = evaluate(model, test_loader, criterion)
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["test_loss"].append(v_loss)
        history["test_acc"].append(v_acc)

        print(f"    Epoch [{epoch:02d}/{num_epochs:02d}] "
              f"Train Loss: {t_loss:.4f} | Train Acc: {t_acc:6.2f}% | "
              f"Val Loss: {v_loss:.4f} | Val Acc: {v_acc:6.2f}%", flush=True)

    elapsed = time.time() - start_time
    print(f"[+] Training completed in {elapsed:.2f} seconds.", flush=True)

    # 4. Final Evaluation
    test_loss, test_acc, y_true, y_pred, y_prob, raw_images = evaluate(model, test_loader, criterion)

    # 5. Metrics & Confusion Matrix
    cm_raw = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
    report = classification_report(y_true, y_pred, target_names=CIFAR10_CLASSES, output_dict=True)

    # 6. Render Visualization Dashboard
    plot_vgg16_dashboard(
        history=history,
        cm_norm=cm_norm,
        report=report,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        raw_images=raw_images,
        output_path="vgg16_cifar10_results.png"
    )

    # 7. Summary Table
    print("\n" + "=" * 80)
    print("                    PER-CLASS CLASSIFICATION REPORT                             ")
    print("=" * 80)
    print(f"{'Class Name':<14} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 80)
    for cls in CIFAR10_CLASSES:
        p = report[cls]["precision"] * 100.0
        r = report[cls]["recall"] * 100.0
        f1 = report[cls]["f1-score"] * 100.0
        sup = report[cls]["support"]
        print(f"{cls:<14} | {p:6.2f}%    | {r:6.2f}%    | {f1:6.2f}%    | {sup:<8}")
    print("=" * 80)
    print(f"Final Validation Accuracy: {test_acc:6.2f}%")
    print(f"Macro F1-Score:            {report['macro avg']['f1-score'] * 100:6.2f}%")
    print(f"Weighted F1-Score:         {report['weighted avg']['f1-score'] * 100:6.2f}%")
    print("=" * 80)
    print("[✓] Execution complete. Dashboard saved to 'vgg16_cifar10_results.png'.\n", flush=True)


if __name__ == "__main__":
    main()

