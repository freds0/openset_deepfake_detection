"""Fast, offline smoke tests for the OSDFD components.

Uses a tiny, randomly-initialised SigLIP vision tower (no checkpoint download)
so the whole model can be exercised on CPU in seconds. Verifies:
  * forward / backward pass and shapes,
  * only PEFT + head params are trainable (frozen backbone),
  * FSM is train-only and mixes only fake samples,
  * SCL / combined loss behave sanely,
  * the metric suite returns the expected keys.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.transforms import build_transform
from src.inference.predictor import OSDFDPredictor
from src.losses.combined import OSDFDLoss
from src.losses.single_center_loss import SingleCenterLoss
from src.models.fsm import ForgeryStyleMixture
from src.models.osdfd import OSDFDModel
from src.models.peft_inject import CDCConfig, LoRAConfig
from src.training.metrics import compute_metrics

# Tiny config: 4 layers, 96-dim, patch16, 64px -> (64/16)^2 = 16 square tokens.
TINY = dict(
    hidden_size=96,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=192,
    patch_size=16,
    image_size=64,
)


def _tiny_model(fsm_prob: float = 0.5) -> OSDFDModel:
    return OSDFDModel(
        lora=LoRAConfig(r=4, alpha=4.0),
        cdc=CDCConfig(bottleneck=16),
        fsm_prob=fsm_prob,
        feature_fusion="global_local",
        head_hidden_dim=32,
        freeze_backbone=True,
        pretrained=False,
        backbone_config_overrides=TINY,
    )


def test_forward_backward_and_trainable_params():
    model = _tiny_model()
    model.train()
    x = torch.randn(6, 3, 64, 64)
    is_fake = torch.tensor([0, 1, 1, 0, 1, 1]).bool()
    domains = torch.tensor([0, 1, 2, 0, 3, 4])

    out = model(x, is_fake=is_fake, domains=domains, apply_fsm=True)
    assert out.logits.shape == (6,)
    assert out.scl_features.shape[0] == 6
    assert out.pooled.shape == (6, 96)

    loss = OSDFDLoss()(out.logits, out.scl_features, is_fake.long())[0]
    loss.backward()

    # Backbone frozen: no grad on vision_model weights; PEFT/head get grads.
    for name, p in model.named_parameters():
        is_peft_or_head = ("lora_" in name) or ("adapter" in name) or name.startswith("head.")
        if is_peft_or_head:
            assert p.requires_grad, f"{name} should be trainable"
        if "vision_model" in name and "lora_" not in name and "adapter" not in name:
            assert not p.requires_grad, f"{name} should be frozen"

    n_train = model.num_trainable_parameters()
    n_total = model.num_total_parameters()
    assert 0 < n_train < n_total


def test_fsm_train_only_and_fake_only():
    fsm = ForgeryStyleMixture(prob=1.0)
    tokens = torch.randn(4, 16, 96)
    is_fake = torch.tensor([0, 1, 0, 1]).bool()
    domains = torch.tensor([0, 1, 0, 2])

    # eval mode: identity
    fsm.eval()
    assert torch.equal(fsm(tokens, is_fake, domains), tokens)

    # train mode: real rows unchanged, at least one fake row changed
    fsm.train()
    torch.manual_seed(0)
    out = fsm(tokens, is_fake, domains)
    assert torch.equal(out[0], tokens[0])  # real
    assert torch.equal(out[2], tokens[2])  # real
    changed = (~torch.isclose(out, tokens).all(dim=(1, 2)))
    assert changed[1] or changed[3]  # some fake row mixed


def test_fsm_single_domain_fallback():
    tokens = torch.randn(6, 16, 96)
    is_fake = torch.tensor([0, 0, 1, 1, 1, 1]).bool()
    domains = torch.tensor([0, 0, 1, 1, 1, 1])  # all fakes share one domain (e.g. NTIRE)

    # "off" reproduces the original no-op behaviour (identity permutation).
    fsm_off = ForgeryStyleMixture(prob=1.0, single_domain_fallback="off")
    fsm_off.train()
    assert torch.equal(fsm_off(tokens, is_fake, domains), tokens)
    assert fsm_off.last_fired is False

    # "random" (default) still mixes: pairs fakes with a random other fake.
    fsm_random = ForgeryStyleMixture(prob=1.0, single_domain_fallback="random")
    fsm_random.train()
    torch.manual_seed(0)
    out = fsm_random(tokens, is_fake, domains)
    assert torch.equal(out[0], tokens[0])  # real rows untouched
    assert torch.equal(out[1], tokens[1])
    changed = ~torch.isclose(out, tokens).all(dim=(1, 2))
    assert changed[2:].any()  # at least one fake row was mixed
    assert fsm_random.last_fired is True

    with pytest.raises(ValueError):
        ForgeryStyleMixture(single_domain_fallback="bogus")


def test_single_center_loss_pushes_fakes_away():
    scl = SingleCenterLoss(margin=0.01)
    torch.manual_seed(0)
    real = torch.randn(8, 16) * 0.5      # some spread -> Dist_R > 0
    labels = torch.tensor([0] * 8 + [1] * 8)
    # fakes far from the real center -> hinge inactive -> loss == Dist_R
    good = scl(torch.cat([real, torch.ones(8, 16) * 5.0]), labels)
    # fakes at the real center -> hinge active -> larger loss
    bad = scl(torch.cat([real, real.mean(0, keepdim=True).repeat(8, 1)]), labels)
    assert bad > good


def test_metrics_keys():
    labels = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.6])
    m = compute_metrics(labels, scores)
    for k in ("acc", "auc", "f1", "ap", "eer", "fpr", "fnr"):
        assert k in m


def test_inference_disables_fsm():
    model = _tiny_model()
    model.eval()
    x = torch.randn(3, 3, 64, 64)
    with torch.no_grad():
        a = model(x, apply_fsm=False).logits
        b = model(x, apply_fsm=False).logits
    assert torch.allclose(a, b)


class _FakeLightningModule:
    """Stand-in for OSDFDLightningModule exposing only what predictor.py uses."""

    def __init__(self, model: OSDFDModel) -> None:
        self.model = model


def test_predict_folder_batched(tmp_path):
    model = _tiny_model(fsm_prob=0.0)
    model.eval()

    predictor = object.__new__(OSDFDPredictor)  # bypass ckpt-loading __init__
    predictor.device = torch.device("cpu")
    predictor.module = _FakeLightningModule(model)
    predictor.transform = build_transform(64, train=False)
    predictor.cropper = None
    predictor.threshold = 0.5

    paths = []
    for i in range(5):
        p = tmp_path / f"img_{i}.png"
        arr = (np.random.RandomState(i).rand(80, 80, 3) * 255).astype(np.uint8)
        Image.fromarray(arr).save(p)
        paths.append(str(p))

    single = [predictor.predict_image(p) for p in sorted(paths)]
    batched = predictor.predict_folder(str(tmp_path), batch_size=2, num_workers=0)

    assert len(batched) == len(single)
    for a, b in zip(single, batched):
        assert a.path == b.path
        assert abs(a.probability - b.probability) < 1e-5
        assert a.label == b.label


@pytest.mark.parametrize("resize_mode", ["squash", "crop"])
@pytest.mark.parametrize("train", [True, False])
def test_build_transform_resize_mode_shape(resize_mode, train):
    image = Image.fromarray((np.random.rand(480, 640, 3) * 255).astype("uint8"))
    transform = build_transform(224, train=train, resize_mode=resize_mode)
    out = transform(image)
    assert out.shape == (3, 224, 224)


def test_build_transform_invalid_resize_mode():
    with pytest.raises(ValueError):
        build_transform(224, resize_mode="bogus")


def test_balance_sampler_yields_balanced_batches(tmp_path):
    from src.data.datamodule import ForgeryDataModule

    n_real, n_fake = 20, 80  # deliberately imbalanced, like NTIRE/FF++
    rows = []
    for i in range(n_real + n_fake):
        p = tmp_path / f"img_{i}.png"
        arr = (np.random.RandomState(i).rand(32, 32, 3) * 255).astype(np.uint8)
        Image.fromarray(arr).save(p)
        label = 0 if i < n_real else 1
        rows.append({"path": str(p), "label": label, "domain": label, "split": "train"})
    import pandas as pd

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    dm = ForgeryDataModule(
        source="manifest",
        manifest=str(manifest_path),
        image_size=32,
        batch_size=8,
        num_workers=0,
        real_oversample=1,
        balance_sampler=True,
        persistent_workers=False,
    )
    dm.setup(stage="fit")
    assert len(dm._train) == n_real + n_fake  # not inflated by duplication

    torch.manual_seed(0)
    loader = dm.train_dataloader()
    labels = []
    for i, batch in enumerate(loader):
        labels.append(batch["label"])
        if i >= 49:
            break
    labels = torch.cat(labels).float()
    fake_ratio = labels.mean().item()
    assert 0.45 <= fake_ratio <= 0.55


def test_balance_sampler_rejects_oversample():
    from src.data.datamodule import ForgeryDataModule

    with pytest.raises(ValueError):
        ForgeryDataModule(balance_sampler=True, real_oversample=4)


def test_train_pool_head_flag():
    base = _tiny_model()
    with_head = OSDFDModel(
        lora=LoRAConfig(r=4, alpha=4.0),
        cdc=CDCConfig(bottleneck=16),
        fsm_prob=0.5,
        feature_fusion="global_local",
        head_hidden_dim=32,
        freeze_backbone=True,
        train_pool_head=True,
        pretrained=False,
        backbone_config_overrides=TINY,
    )
    head_params = sum(p.numel() for p in base.backbone.vision_model.head.parameters())
    assert head_params > 0
    assert (
        with_head.num_trainable_parameters() - base.num_trainable_parameters() == head_params
    )


def test_ema_callback_decay():
    from src.utils.ema import EMACallback

    class _TinyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor([1.0]))

    module = _TinyModule()
    ema = EMACallback(decay=0.9)
    ema.on_fit_start(trainer=None, pl_module=module)
    assert torch.allclose(ema._shadow["w"], torch.tensor([1.0]))

    expected = 1.0
    for new_val in (2.0, 3.0, 4.0):
        with torch.no_grad():
            module.w.fill_(new_val)
        ema.on_train_batch_end(None, module, None, None, 0)
        expected = 0.9 * expected + 0.1 * new_val
        assert torch.allclose(ema._shadow["w"], torch.tensor([expected]), atol=1e-6)

    original = module.w.detach().clone()
    ema._swap_in(module)
    assert torch.allclose(module.w, ema._shadow["w"])
    ema._swap_out(module)
    assert torch.allclose(module.w, original)


def test_scl_margin_scale():
    D = 16
    real = torch.zeros(4, D)
    fake = torch.zeros(4, D)  # fakes exactly at the real center -> dist_r = dist_f = 0
    features = torch.cat([real, fake])
    labels = torch.tensor([0] * 4 + [1] * 4)

    loss_none = SingleCenterLoss(margin=0.3, margin_scale="none")(features, labels)
    assert torch.allclose(loss_none, torch.tensor(0.3), atol=1e-6)

    loss_sqrt = SingleCenterLoss(margin=0.3, margin_scale="sqrt_dim")(features, labels)
    assert torch.allclose(loss_sqrt, torch.tensor(0.3 * 4.0), atol=1e-6)  # 0.3*sqrt(16)

    with pytest.raises(ValueError):
        SingleCenterLoss(margin_scale="bogus")


def test_torchmetrics_auc_matches_sklearn():
    from torchmetrics.classification import BinaryAUROC

    labels = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.3, 0.6])
    sklearn_auc = compute_metrics(labels, scores)["auc"]

    metric = BinaryAUROC()
    metric.update(torch.tensor(scores), torch.tensor(labels))
    tm_auc = metric.compute().item()
    assert abs(sklearn_auc - tm_auc) < 1e-6


def test_calibrated_threshold_roundtrip():
    from src.lightning.module import OSDFDLightningModule

    module = object.__new__(OSDFDLightningModule)  # bypass __init__ (no Hydra cfg needed)
    module.calibrated_threshold = 0.37
    ckpt: dict = {}
    module.on_save_checkpoint(ckpt)
    assert ckpt["calibrated_threshold"] == 0.37

    fresh = object.__new__(OSDFDLightningModule)
    fresh.on_load_checkpoint(ckpt)
    assert fresh.calibrated_threshold == 0.37

    # Older checkpoints without the key fall back to the un-calibrated default.
    fresh_old = object.__new__(OSDFDLightningModule)
    fresh_old.on_load_checkpoint({})
    assert fresh_old.calibrated_threshold == 0.5


def test_fsm_domain_shuffle_vectorized_pairs():
    fsm = ForgeryStyleMixture(prob=1.0)
    domains = torch.tensor([1, 1, 2, 2, 3])
    for _ in range(200):
        perm = fsm._domain_shuffle(domains)
        assert (domains[perm] != domains).all()


def test_peft_start_layer_truncates_injection():
    base = _tiny_model()  # start_layer=0: all 4 tiny blocks adapted
    partial = OSDFDModel(
        lora=LoRAConfig(r=4, alpha=4.0),
        cdc=CDCConfig(bottleneck=16),
        fsm_prob=0.5,
        feature_fusion="global_local",
        head_hidden_dim=32,
        freeze_backbone=True,
        peft_start_layer=2,
        pretrained=False,
        backbone_config_overrides=TINY,
    )
    # Exactly half of the 4 blocks adapted -> per-layer PEFT params halve.
    def peft_params(m):
        return sum(
            p.numel() for n, p in m.named_parameters()
            if p.requires_grad and not n.startswith("head.")
        )

    assert peft_params(partial) == peft_params(base) // 2

    # Blocks below start_layer are untouched (no LoRA/Adapter wrappers).
    from src.models.cdc_adapter import AdapterMLP
    from src.models.lora import LoRALinear

    layers = partial.backbone.encoder_layers
    assert not isinstance(layers[0].self_attn.q_proj, LoRALinear)
    assert not isinstance(layers[1].mlp, AdapterMLP)
    assert isinstance(layers[2].self_attn.q_proj, LoRALinear)
    assert isinstance(layers[3].mlp, AdapterMLP)

    # Forward/backward still work end-to-end.
    partial.train()
    out = partial(torch.randn(2, 3, 64, 64), apply_fsm=False)
    out.logits.sum().backward()


def test_forgery_frame_dataset_jpeg_draft(tmp_path):
    from src.data.dataset import ForgeryFrameDataset, Record

    arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    jpeg_path = tmp_path / "a.jpg"
    png_path = tmp_path / "b.png"
    Image.fromarray(arr).save(jpeg_path, format="JPEG")
    Image.fromarray(arr).save(png_path, format="PNG")

    records = [
        Record(path=str(jpeg_path), label=0, domain=0),
        Record(path=str(png_path), label=0, domain=0),
    ]
    ds = ForgeryFrameDataset(records, transform=build_transform(32, train=False), jpeg_draft_size=16)
    for i in range(len(ds)):
        assert ds[i]["pixel_values"].shape == (3, 32, 32)


def _write_ffpp_tree(root, split, classes, n=2):
    """Create a tiny FF++-style split tree: <root>/<split>/<class>/*.png."""
    import os

    for cls in classes:
        d = os.path.join(root, split, cls)
        os.makedirs(d, exist_ok=True)
        for i in range(n):
            arr = (np.random.RandomState(hash((cls, i)) % 2**32).rand(16, 16, 3) * 255).astype(
                np.uint8
            )
            Image.fromarray(arr).save(os.path.join(d, f"{i}.png"))


def test_records_from_faceforensics_class_filter(tmp_path):
    from src.data.dataset import records_from_faceforensics

    root = str(tmp_path)
    all_classes = ["real", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    _write_ffpp_tree(root, "train", all_classes)

    # No filter -> every manipulation is present.
    full = records_from_faceforensics(root, "train")
    assert {r.domain for r in full} == {0, 1, 2, 3, 4}

    # Source-domain subset (leave-out Deepfakes): DF domain (1) must be absent.
    src = records_from_faceforensics(
        root, "train", classes=["real", "Face2Face", "FaceSwap", "NeuralTextures"]
    )
    domains = {r.domain for r in src}
    assert 1 not in domains and domains == {0, 2, 3, 4}

    # An empty match is a configuration error, not a silent no-op.
    with pytest.raises(ValueError):
        records_from_faceforensics(root, "train", classes=["does_not_exist"])


def test_datamodule_loo_train_test_class_split(tmp_path):
    from src.data.datamodule import ForgeryDataModule

    root = str(tmp_path)
    all_classes = ["real", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    for split in ("train", "val", "test"):
        _write_ffpp_tree(root, split, all_classes)

    dm = ForgeryDataModule(
        source="faceforensics",
        root=root,
        image_size=16,
        batch_size=4,
        num_workers=0,
        real_oversample=1,
        persistent_workers=False,
        train_classes=["real", "Face2Face", "FaceSwap", "NeuralTextures"],
        test_classes=["real", "Deepfakes"],
    )
    dm.setup(stage="fit")
    dm.setup(stage="test")

    # Train + val: source manipulations only (Deepfakes / domain 1 excluded).
    assert 1 not in {r.domain for r in dm._train.records}
    assert 1 not in {r.domain for r in dm._val.records}
    # Test: held-out target manipulation (Deepfakes) + real only.
    assert {r.domain for r in dm._test.records} == {0, 1}
