"""
Utilitários de dados: MNIST, FashionMNIST, Split-MNIST.
"""
from __future__ import annotations
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T


def _num_workers() -> int:
    """2 workers no Colab/Linux, 0 no Windows (evita erros de multiprocessing)."""
    import platform
    return 2 if platform.system() != "Windows" else 0


def get_mnist(batch_size: int = 512, data_dir: str = "./data",
              train_subset: int | None = None, val_subset: int | None = None) -> tuple[DataLoader, DataLoader]:
    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    train = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    if train_subset:
        train = Subset(train, list(range(min(train_subset, len(train)))))
    if val_subset:
        test = Subset(test, list(range(min(val_subset, len(test)))))
    nw = _num_workers()
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True),
        DataLoader(test, batch_size=1024, shuffle=False, num_workers=nw, pin_memory=True),
    )


def get_fashion_mnist(batch_size: int = 512, data_dir: str = "./data",
                      train_subset: int | None = None, val_subset: int | None = None) -> tuple[DataLoader, DataLoader]:
    transform = T.Compose([T.ToTensor(), T.Normalize((0.2860,), (0.3530,))])
    train = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
    test = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    if train_subset:
        train = Subset(train, list(range(min(train_subset, len(train)))))
    if val_subset:
        test = Subset(test, list(range(min(val_subset, len(test)))))
    nw = _num_workers()
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True),
        DataLoader(test, batch_size=1024, shuffle=False, num_workers=nw, pin_memory=True),
    )


def get_split_mnist(
    task_id: int,
    batch_size: int = 256,
    data_dir: str = "./data",
    train_subset: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Split-MNIST: 5 tarefas binárias.
    task_id 0 → dígitos (0,1), 1 → (2,3), ..., 4 → (8,9)
    """
    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    train_full = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_full = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    labels = [task_id * 2, task_id * 2 + 1]

    def filter_dataset(ds):
        targets = ds.targets if hasattr(ds, "targets") else torch.tensor(ds.labels)
        idx = [i for i, t in enumerate(targets) if int(t) in labels]
        return Subset(ds, idx)

    train_sub = filter_dataset(train_full)
    test_sub = filter_dataset(test_full)
    if train_subset:
        train_sub = Subset(train_sub, list(range(min(train_subset, len(train_sub)))))
    nw = _num_workers()
    return (
        DataLoader(train_sub, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True),
        DataLoader(test_sub, batch_size=1024, shuffle=False, num_workers=nw, pin_memory=True),
    )
