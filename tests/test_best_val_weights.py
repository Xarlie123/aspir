"""Keep the best-validation weights after training (Single Test and Batch Test).

Self-contained script in the style of tests/test_solver_parameters.py:

    .venv/bin/python tests/test_best_val_weights.py

ASPIR used to end training with the last epoch's weights. With a constant
learning rate the last epoch can be several dB below the best one, so the option
restores the weights of the epoch with the lowest validation loss. Off by
default: configurations saved before the option keep their behaviour.
"""

import logging
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import torch

from simulation_engine._4_postprocessor.postprocessor_nn import PostprocessorNN
from ui.custom_widgets.batch_test.test_config_model import TestConfiguration

LOGGER = logging.getLogger("test_best_val")
EPOCHS = 8


def _postprocessor(val_ratio=0.2, seed=0):
    """A small U-Net on 16x16 blurred-noise pairs, on the CPU so losses are reproducible."""
    rng = np.random.default_rng(seed)
    clean = rng.random((60, 16, 16)).astype(np.float32)
    noisy = clean + 0.3 * rng.standard_normal(clean.shape).astype(np.float32)
    dataset = SimpleNamespace(data=list(clean), img_size=16, seed=10)
    torch.manual_seed(seed)
    return PostprocessorNN(
        "u-net", {"img_size": 16}, dataset, applicator=None, batch_size=8, lr=3e-2,
        use_gpu=False, train_ratio=0.6, val_ratio=val_ratio, test_ratio=1 - 0.6 - val_ratio,
        logger=LOGGER, reconstructed_data=noisy,
    )


# Validation losses scripted so the best epoch (3) is neither the first nor the last
SCRIPTED_VAL = [0.9, 0.5, 0.2, 0.4, 0.6, 0.3, 0.7, 0.8]
BEST_EPOCH = 3


def _scripted(pp):
    """Script validate() and snapshot the weights each epoch (taken before validation)."""
    losses, snapshots = iter(SCRIPTED_VAL), {}
    pp.validate = lambda: next(losses)

    def snapshot(epoch, _total):
        snapshots[epoch] = {k: v.detach().clone() for k, v in pp.model.state_dict().items()}
    return snapshot, snapshots


def _same(state, snapshot):
    return all(torch.equal(v, snapshot[k]) for k, v in state.items())


def test_best_validation_weights_are_restored():
    """Single Test path (train_with_metrics): the model ends with epoch 3's weights."""
    pp, curves = _postprocessor(), {}
    snapshot, snapshots = _scripted(pp)
    pp.train_with_metrics(EPOCHS, progress_callback=snapshot,
                          metrics_callback=lambda v, *_: curves.update(val=v), keep_best_val=True)
    assert curves["val"] == SCRIPTED_VAL, curves["val"]
    assert (pp.kept_epoch, pp.kept_weights) == (BEST_EPOCH, "best_validation"), pp.kept_epoch
    state = pp.model.state_dict()
    assert _same(state, snapshots[BEST_EPOCH]), "weights are not those of the best epoch"
    assert not _same(state, snapshots[EPOCHS]), "weights still those of the last epoch"
    print(f"  epoch {pp.kept_epoch}/{EPOCHS} restored exactly (BatchNorm buffers included)")


def test_default_keeps_the_last_epoch():
    pp = _postprocessor()
    snapshot, snapshots = _scripted(pp)
    pp.train_with_metrics(EPOCHS, progress_callback=snapshot)
    assert (pp.kept_epoch, pp.kept_weights) == (EPOCHS, "last"), (pp.kept_epoch, pp.kept_weights)
    assert _same(pp.model.state_dict(), snapshots[EPOCHS])
    print("  last epoch kept, as before the option")


def test_train_without_metrics_also_selects():
    """Batch Test without the training-curves report trains through train()."""
    pp = _postprocessor()
    snapshot, snapshots = _scripted(pp)
    pp.train(EPOCHS, progress_callback=snapshot, keep_best_val=True)
    assert (pp.kept_epoch, pp.kept_weights) == (BEST_EPOCH, "best_validation"), pp.kept_epoch
    assert _same(pp.model.state_dict(), snapshots[BEST_EPOCH])
    print(f"  train(): epoch {pp.kept_epoch} restored")


def test_real_validation_loss_matches_the_kept_epoch():
    """Unscripted: after training, validating the model gives the minimum of the curve."""
    pp, curves = _postprocessor(), {}
    pp.train_with_metrics(4, metrics_callback=lambda v, *_: curves.update(val=v), keep_best_val=True)
    best = int(np.argmin(curves["val"]))
    assert pp.kept_epoch == best + 1, (pp.kept_epoch, best + 1)
    assert abs(pp.validate() - curves["val"][best]) < 1e-6, (pp.validate(), curves["val"][best])
    print(f"  val loss after training {pp.validate():.5f} = curve minimum (epoch {pp.kept_epoch})")


def test_empty_validation_split_falls_back_to_last():
    pp = _postprocessor(val_ratio=0.0)
    pp.train(2, keep_best_val=True)
    assert (pp.kept_epoch, pp.kept_weights) == (2, "last"), (pp.kept_epoch, pp.kept_weights)
    print("  no validation images: last epoch kept, with a warning")


def test_batch_config_round_trip():
    assert TestConfiguration().keep_best_val is False
    cfg = TestConfiguration(keep_best_val=True)
    assert TestConfiguration.from_dict(cfg.to_dict()).keep_best_val is True
    assert TestConfiguration.from_dict({}).keep_best_val is False
    print("  saved and reloaded; omitted key -> off")


def test_gui_controls():
    from PySide6.QtWidgets import QApplication

    from ui.custom_widgets.batch_test.test_config_widget import TestConfigWidget
    from ui.custom_widgets.postprocessor_control.nn_control.nn_control_widget import NNControlWidget

    _app = QApplication.instance() or QApplication([])

    nn = NNControlWidget(logger=LOGGER)
    emitted = []
    nn.trainRequested.connect(lambda *args: emitted.append(args))
    assert not nn.keep_best_val_checkbox.isChecked()
    nn.keep_best_val_checkbox.setChecked(True)
    nn.train_button.click()
    assert emitted and emitted[-1][-1] is True, emitted

    widget = TestConfigWidget(logger=LOGGER)
    cfg = TestConfiguration(keep_best_val=True)
    widget.set_config(cfg)
    assert widget.keep_best_val_checkbox.isChecked()
    widget.keep_best_val_checkbox.setChecked(False)
    assert cfg.keep_best_val is False
    print("  Single Test emits the option; Batch Test shows and edits it")


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failures = 0
    for fn in tests:
        print(f"\n{fn.__name__}")
        try:
            fn()
            print("  PASS")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
