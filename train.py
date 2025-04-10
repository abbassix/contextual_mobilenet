import yaml
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from models import MobileNetV2, ContextualMobileNetV2
from utils import set_seed, accuracy_topk, get_dataloaders, init_wandb, count_parameters


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_model(cfg):
    if cfg["model"] == "baseline":
        model = MobileNetV2(num_classes=cfg["num_classes"], width_mult=cfg["width_mult"])
    elif cfg["model"] == "contextual":
        model = ContextualMobileNetV2(num_classes=cfg["num_classes"], width_mult=cfg["width_mult"])
    else:
        raise ValueError(f"Unknown model type: {cfg['model']}")
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        acc1, _ = accuracy_topk(out, y, topk=(1, 5))
        total_correct += acc1 * x.size(0)

    return total_loss / len(loader.dataset), total_correct / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, top1_total, top5_total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        top1, top5 = accuracy_topk(out, y, topk=(1, 5))

        total_loss += loss.item() * x.size(0)
        top1_total += top1 * x.size(0)
        top5_total += top5 * x.size(0)

    size = len(loader.dataset)
    return total_loss / size, top1_total / size, top5_total / size


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])

    train_loader, val_loader = get_dataloaders(cfg["batch_size"], cfg["num_workers"])
    model = get_model(cfg).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg["learning_rate"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"]
    )

    if cfg["scheduler"] == "step":
        scheduler = StepLR(optimizer, step_size=cfg["step_size"], gamma=cfg["gamma"])
    elif cfg["scheduler"] == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    else:
        scheduler = None

    init_wandb(cfg)
    wandb.watch(model)
    wandb.config.update({"params": count_parameters(model)})

    for epoch in range(cfg["epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc1, val_acc5 = evaluate(model, val_loader, criterion, device)
        if scheduler:
            scheduler.step()

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/acc@1": train_acc,
            "val/loss": val_loss,
            "val/acc@1": val_acc1,
            "val/acc@5": val_acc5,
            "lr": optimizer.param_groups[0]["lr"]
        })

        print(f"[{epoch+1:03d}] Train Loss: {train_loss:.4f}, Acc@1: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc@1: {val_acc1:.4f}, Acc@5: {val_acc5:.4f}")

    wandb.finish()


if __name__ == "__main__":
    main()
