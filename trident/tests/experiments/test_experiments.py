import torch
from hydra import compose, initialize

from trident.train import train

# 2. The module with your configs should be importable.
#    it needs to have a __init__.py (can be empty).
# 3. THe config path is relative to the file calling initialize (this file)


def test_single_train_dataloader() -> None:
    with initialize(
        version_base=None,
        config_path="../../configs/tests/",
    ):
        # config is relative to a module
        cfg = compose(config_name="test_single_train_single_val_test")
        assert torch.allclose(torch.zeros(1)[0], torch.Tensor(train(cfg)))


def test_multi_train_dataloader() -> None:
    with initialize(
        version_base=None,
        config_path="../../configs/tests/",
    ):
        # config is relative to a module
        cfg = compose(config_name="test_many_train_single_val_test")
        assert torch.allclose(torch.zeros(1)[0], torch.Tensor(train(cfg)))


def test_off_by_one() -> None:
    with initialize(
        version_base=None,
        config_path="../../configs/tests/",
    ):
        # config is relative to a module
        cfg = compose(config_name="test_single_train_many_val_test")
        assert torch.allclose(torch.Tensor([1]), torch.Tensor(train(cfg)))


def test_off_by_two() -> None:
    with initialize(
        version_base=None,
        config_path="../../configs/tests/",
    ):
        # config is relative to a module
        cfg = compose(
            config_name="test_single_train_many_val_test",
            overrides=["optimized_metric='off_by_two/val/mse_loss'"],
        )
        assert torch.allclose(torch.Tensor([4]), torch.Tensor(train(cfg)))
