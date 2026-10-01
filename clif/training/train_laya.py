"""Fine-tunes a Laya checkpoint on SNOMED choice data (from ``clif export-laya-training``).

Single-device port of the training loop in Laya's fine-tuning notebook
(notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb, v0.3.21): the same
objective (policy gradient against Laya's proper scoring reward plus soft
cross-entropy), the same learning rates for encoder and head, and the same
post-training temperature fit on a held-out slice. Differences:

- runs on one device (Apple GPU via MPS, CUDA or CPU) instead of 2x T4 DDP;
- every example is encoded with the ``head_max_len`` / ``max_len`` stored in its
  row, i.e. exactly the budget Clif sends at inference time (the notebook uses
  one fixed budget);
- examples are grouped by length to cut padding;
- a validation accuracy is reported after every epoch.

The output directory is a regular Laya checkpoint: ``laya.load(path)`` works and
``CLIF_LAYA_CHECKPOINT=path`` with ``CLIF_RERANKER=laya-local`` uses it in Clif.

    clif train-laya --data data/intl/laya_train.jsonl --out models/laya-snomed
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

QUESTION_ID = "snomed_concept"


@dataclass
class TrainConfig:
    data: Path
    out: Path
    base: str = "convaiinnovations/laya"
    subfolder: str = "multilingual"
    epochs: int = 2
    micro_batch: int = 8
    grad_accum: int = 4
    group_size: int = 4
    lr_encoder: float = 2.5e-5
    lr_head: float = 1.0e-4
    sigma_start: float = 0.4
    sigma_end: float = 0.1
    max_items: int = 0
    holdout: float = 0.05
    holdout_max: int = 3000
    device: str = ""
    seed: int = 20260930
    save_every: int = 5000  # optimizer steps between rolling checkpoints (0 = off)
    resume: bool = False


def _resolve_base(base: str, subfolder: str) -> str:
    if os.path.isdir(base):
        return os.path.join(base, subfolder) if subfolder and os.path.isdir(os.path.join(base, subfolder)) else base
    from huggingface_hub import snapshot_download

    prefix = f"{subfolder}/" if subfolder else ""
    patterns = [f"{prefix}rl_agent_config.json", f"{prefix}model.safetensors", f"{prefix}tokenizer/*",
                f"{prefix}encoder/*"]
    root = snapshot_download(base, allow_patterns=patterns)
    return os.path.join(root, subfolder) if subfolder else root


def _pick_device(requested: str):
    import torch

    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_items(path: Path, tok, max_items: int = 0) -> list[dict]:
    import numpy as np

    from laya.common import QTYPES, build_sequence, render_options

    items: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            state = json.loads(row["state"])
            q = json.loads(row["questions"])[QUESTION_ID]
            gold = json.loads(row["gold"])[QUESTION_ID]
            crit = q["criteria"]
            keys = list(crit.keys())
            target = [float(gold["probabilities"].get(k, 0.0)) for k in keys]
            s = sum(target)
            if s <= 0:
                continue
            target = [v / s for v in target]
            internal = {"t": "choice", "ins": q["instructions"], "crit": crit}
            seq, markers = build_sequence(tok, state, internal, row["max_len"], row["head_max_len"])
            if len(markers) != len(render_options(internal)):
                continue  # an option did not fit; skip rather than train on a truncated question
            # int32 arrays instead of Python lists: ~7x less memory for the full SNOMED set
            items.append({"ids": np.asarray(seq, dtype=np.int32), "markers": markers, "qtype": QTYPES["choice"],
                          "target": target,
                          "label": target.index(max(target)), "kind": row.get("meta", {}).get("kind", "")})
            if max_items and len(items) >= max_items:
                break
            if len(items) % 50000 == 0:
                log.info("  tokenized %d examples", len(items))
    return items


def _collate(items: list[dict], pad_id: int):
    import torch

    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.as_tensor(it["ids"], dtype=torch.long)
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, :k] = torch.tensor(it["target"], dtype=torch.float32)
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask, "target": target,
            "qtype": torch.tensor([it["qtype"] for it in items]), "label": torch.tensor([it["label"] for it in items])}


def _batches(items: list[dict], size: int, rng: random.Random) -> list[list[dict]]:
    """Length-grouped batches in random order: similar lengths share a batch, less padding."""
    order = sorted(items, key=lambda it: len(it["ids"]) + rng.random() * 8)
    batches = [order[i : i + size] for i in range(0, len(order), size)]
    rng.shuffle(batches)
    return batches


def _fit_temperature(pairs: list[tuple[list[float], list[float]]]) -> float:
    import torch

    if len(pairs) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in pairs)
    Z = torch.full((len(pairs), kmax), -1e4)
    T = torch.zeros((len(pairs), kmax))
    for i, (z, t) in enumerate(pairs):
        Z[i, : len(z)] = torch.tensor(z)
        T[i, : len(t)] = torch.tensor(t)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def _evaluate(model, items: list[dict], pad_id: int, device, batch: int = 16) -> tuple[float, list]:
    import torch

    model.eval()
    correct, preds = 0, []
    with torch.no_grad():
        for i in range(0, len(items), batch):
            chunk = items[i : i + batch]
            b = _collate(chunk, pad_id)
            logits, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["marker_pos"].to(device),
                              b["marker_mask"].to(device), b["qtype"].to(device))
            logits = logits.float().cpu()
            for r, it in enumerate(chunk):
                k = len(it["markers"])
                z = logits[r, :k].tolist()
                correct += int(max(range(k), key=lambda j: z[j]) == it["label"])
                preds.append((z, it["target"]))
    model.train()
    return correct / max(1, len(items)), preds


def _save_model(model, tok, mcfg: dict, out: Path) -> None:
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().to("cpu").contiguous() for k, v in model.state_dict().items()},
              str(out / "model.safetensors"))
    model.encoder.config.save_pretrained(str(out / "encoder"))
    tok.save_pretrained(str(out / "tokenizer"))
    (out / "rl_agent_config.json").write_text(json.dumps(mcfg, indent=2))


def train(cfg: TrainConfig) -> dict:
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    from laya.common import build_model, proper_reward

    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    base_dir = _resolve_base(cfg.base, cfg.subfolder)
    with open(os.path.join(base_dir, "rl_agent_config.json")) as f:
        mcfg = json.load(f)
    tok = AutoTokenizer.from_pretrained(os.path.join(base_dir, "tokenizer"))
    model = build_model(mcfg, encoder_dir=os.path.join(base_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(base_dir, "model.safetensors")), strict=True)
    device = _pick_device(cfg.device)
    model.to(device)
    model.train()
    log.info("base %s on %s", base_dir, device)

    items = load_items(cfg.data, tok, cfg.max_items)
    rng.shuffle(items)
    n_hold = min(cfg.holdout_max, max(50, int(len(items) * cfg.holdout)))
    hold, train_items = items[:n_hold], items[n_hold:]
    log.info("%d training items, %d held out", len(train_items), len(hold))
    base_acc, _ = _evaluate(model, hold, tok.pad_token_id, device)
    log.info("held-out accuracy before training: %.3f", base_acc)

    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": cfg.lr_encoder}, {"params": head, "lr": cfg.lr_head}],
                            weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_items) / cfg.micro_batch)
    total_updates = max(1, steps_per_epoch * cfg.epochs // cfg.grad_accum)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_updates, eta_min=1e-6)
    use_bf16 = device.type in ("cuda", "mps")

    ckpt = Path(cfg.out) / "checkpoint_latest"
    start_epoch, skip_steps, history = 0, 0, []
    if cfg.resume and (ckpt / "progress.json").exists():
        prog = json.loads((ckpt / "progress.json").read_text())
        model.load_state_dict(load_file(str(ckpt / "model.safetensors")), strict=True)
        state = torch.load(ckpt / "optimizer.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        start_epoch, skip_steps, history = prog["epoch"], prog["step"], prog.get("history", [])
        log.info("resuming at epoch %d, step %d", start_epoch + 1, skip_steps)

    def save_progress(epoch: int, step: int) -> None:
        _save_model(model, tok, mcfg, ckpt)
        torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict()}, ckpt / "optimizer.pt")
        (ckpt / "progress.json").write_text(json.dumps({"epoch": epoch, "step": step, "history": history}))
        log.info("rolling checkpoint saved (epoch %d, step %d)", epoch + 1, step)

    t0 = time.time()
    for epoch in range(cfg.epochs):
        sigma = cfg.sigma_start + (cfg.sigma_end - cfg.sigma_start) * (epoch / max(1, cfg.epochs - 1))
        batches = _batches(train_items, cfg.micro_batch, rng)  # deterministic per epoch: resume-safe
        if epoch < start_epoch:
            continue
        opt.zero_grad(set_to_none=True)
        run_loss = 0.0
        for step, chunk in enumerate(batches, 1):
            if epoch == start_epoch and step <= skip_steps:
                continue
            b = _collate(chunk, tok.pad_token_id)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits, act = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                                    b["marker_pos"].to(device), b["marker_mask"].to(device), b["qtype"].to(device))
            logits = logits.float()
            mask = b["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = b["target"].to(device)
            qtype = b["qtype"].to(device)
            # Laya's objective: sampled logit perturbations scored by a proper scoring rule
            # (policy gradient, group-relative baseline) plus soft cross-entropy.
            eps = torch.randn((cfg.group_size,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
                adv = (r - r.mean(0, keepdim=True)) / (r.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + loss_ce) / cfg.grad_accum + 0.0 * act.sum()
            loss.backward()
            run_loss += loss_ce.item()
            if step % cfg.grad_accum == 0 or step == len(batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                if cfg.save_every and (step // cfg.grad_accum) % cfg.save_every == 0:
                    save_progress(epoch, step)
            if step % 500 == 0:
                el = time.time() - t0
                done = (epoch - start_epoch) * len(batches) + step - (skip_steps if epoch == start_epoch else 0)
                eta = el / max(1, done) * ((cfg.epochs - epoch) * len(batches) - step)
                log.info("epoch %d step %d/%d  ce %.4f  %.0fs elapsed, ~%.0f min left", epoch + 1, step,
                         len(batches), run_loss / 500, el, eta / 60)
                run_loss = 0.0
        acc, _ = _evaluate(model, hold, tok.pad_token_id, device)
        history.append({"epoch": epoch + 1, "heldout_accuracy": acc, "minutes": (time.time() - t0) / 60})
        log.info("epoch %d done: held-out accuracy %.3f", epoch + 1, acc)
        if cfg.save_every:
            save_progress(epoch + 1, 0)

    acc, preds = _evaluate(model, hold, tok.pad_token_id, device)
    temp = _fit_temperature(preds)
    out = Path(cfg.out)
    base_temps = mcfg.get("temperature", [1.0, 1.0, 1.0])
    mcfg["temperature"] = [temp, base_temps[1], base_temps[2]]  # choice fitted; score/noul untouched
    mcfg.pop("temperature_by_options", None)
    mcfg["model_name"] = "laya-snomed"
    mcfg["fine_tuned"] = True
    mcfg["clif_training"] = {
        "base": base_dir, "data": str(cfg.data), "items": len(train_items), "heldout": len(hold),
        "epochs": cfg.epochs, "heldout_accuracy_before": base_acc, "heldout_accuracy_after": acc,
        "choice_temperature": temp, "history": history, "device": str(device),
        "hours": round((time.time() - t0) / 3600, 2),
    }
    _save_model(model, tok, mcfg, out)
    log.info("saved %s (held-out accuracy %.3f -> %.3f, choice temperature %.2f)", out, base_acc, acc, temp)
    return mcfg["clif_training"]
