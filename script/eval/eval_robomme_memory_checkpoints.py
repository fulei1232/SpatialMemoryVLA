#!/usr/bin/env python3
"""Offline RoboMME comparison for C-memory1 and C-memory16 checkpoints.

RoboMME is an offline action dataset, so this reports diffusion action loss and
an explicitly-labelled loss-threshold success proxy. It also pairs Normal and
temporal-Occlusion records by episode/timestep to measure occlusion degradation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

# The training environment uses a local Llama-2 mirror; avoid requiring gated
# access to meta-llama when evaluating an already materialized checkpoint.
os.environ.setdefault(
    "PRISMATIC_LLAMA2_7B_REPO",
    "/media/fulei/jlu/SpatialMemoryVLA/pretrained/NousResearch-Llama-2-7b-hf",
)

import numpy as np
import torch
from torch.utils.data import DataLoader

from vla import get_vla_dataset_and_collator, load_vla


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("/media/fulei/jlu/SpatialMemoryVLA/datasets/robomme_preprocessed_data"))
    p.add_argument("--manifest", type=Path, default=Path("/media/fulei/jlu/SpatialMemoryVLA/datasets/robomme_memory/manifest_occ_200_len16.npz"))
    p.add_argument("--output-root", type=Path, default=Path("/media/fulei/jlu/SpatialMemoryVLA/evaluations/robomme_memory_500step"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--loss-success-threshold", type=float, default=0.10)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", action="append", required=True, help="label=checkpoint.pt")
    return p.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint(path: Path):
    run_config = json.loads((path.parents[1] / "config.json").read_text())
    # Read memory settings from each run so C1 remains C1 and C16 remains C16.
    return load_vla(
        model_id_or_path=str(path), load_for_training=False,
        action_dim=int(run_config.get("action_dim", 8)),
        future_action_window_size=int(run_config.get("future_action_window_size", 15)),
        action_model_type=run_config.get("action_model_type", "DiT-L"),
        use_bf16=True, dataloader_type=run_config.get("dataloader_type", "stream"),
        group_size=int(run_config.get("group_size", 16)),
        per_token_size=int(run_config.get("per_token_size", 256)),
        mem_length=int(run_config["mem_length"]),
        retrieval_layers=int(run_config.get("retrieval_layers", 2)),
        use_timestep_pe=bool(run_config.get("use_timestep_pe", True)),
        fusion_type=run_config.get("fusion_type", "gate"),
        consolidate_type=run_config.get("consolidate_type", "tome"),
        update_fused=bool(run_config.get("update_fused", False)),
        use_spatial_forcing=False,
        use_spatial_memory=bool(run_config.get("use_spatial_memory", True)),
        spatial_align_layer=int(run_config.get("spatial_align_layer", 24)),
        spatial_align_coeff=float(run_config.get("spatial_align_coeff", 0.5)),
        spatial_teacher_dim=int(run_config.get("spatial_teacher_dim", 2048)),
        spatial_debug_asserts=bool(run_config.get("spatial_debug_asserts", True)),
    ).cuda().eval()


def run_condition(model, checkpoint_label, condition, ns, cfg):
    enabled = condition == "occlusion"
    dataset, _, collator = get_vla_dataset_and_collator(
        data_root_dir=ns.data_root, data_mix="robomme",
        image_transform=model.vision_backbone.get_image_transform(),
        tokenizer=model.llm_backbone.get_tokenizer(),
        prompt_builder_fn=model.llm_backbone.prompt_builder_fn,
        default_image_resolution=model.vision_backbone.default_image_resolution,
        image_aug=False, future_action_window_size=15,
        load_all_data_for_training=True, dataloader_type="stream", group_size=16,
        seed=ns.seed, episode_manifest_path=ns.manifest,
        memory_curriculum_enabled=enabled,
        memory_curriculum_type="occlusion" if enabled else "normal",
        occlusion_probability=0.5, occlusion_start_ratio=0.4,
        occlusion_duration_ratio=0.2, occlusion_strength="full",
    )
    loader = DataLoader(dataset, batch_size=ns.batch_size, collate_fn=collator,
                        num_workers=0, pin_memory=True)
    model.cog_mem_bank.reset(); model.per_mem_bank.reset()
    records = []
    sample_count = 0
    with torch.inference_mode():
        for batch in loader:
            b = batch["input_ids"].shape[0]
            if ns.max_samples and sample_count >= ns.max_samples:
                break
            if ns.max_samples:
                keep = min(b, ns.max_samples - sample_count)
                if keep < b:
                    for key in ("input_ids", "attention_mask", "labels", "actions", "action_masks"):
                        batch[key] = batch[key][:keep]
                    batch["episode_ids"] = batch["episode_ids"][:keep]
                    batch["timesteps"] = batch["timesteps"][:keep]
                    batch["occlusion_flags"] = batch["occlusion_flags"][:keep]
                    batch["occlusion_strengths"] = batch["occlusion_strengths"][:keep]
                    b = keep
            device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype
            pixel_values = batch["pixel_values"]
            if isinstance(pixel_values, dict):
                pixel_values = {k: v.to(device=device, dtype=model_dtype) for k, v in pixel_values.items()}
            else:
                pixel_values = pixel_values.to(device=device, dtype=model_dtype)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    pixel_values=pixel_values,
                    labels=batch["labels"].to(device),
                    actions=batch["actions"].to(device=device, dtype=model_dtype),
                    action_masks=batch["action_masks"].to(device),
                    timesteps=batch["timesteps"], episode_ids=batch["episode_ids"],
                    repeated_diffusion_steps=1,
                    occlusion_flags=batch["occlusion_flags"],
                    occlusion_strengths=batch["occlusion_strengths"],
                )
            losses = output[2].get("action_loss", output[0]).detach().float().cpu().item()
            # The model returns one scalar diffusion loss for the batch. Keep it on each row
            # for pairing, and separately record per-batch diagnostics from the gate bank.
            cog = model.cog_mem_bank.last_gate_records
            per = model.per_mem_bank.last_gate_records
            for i in range(b):
                eid = int(batch["episode_ids"][i]); ts = int(batch["timesteps"][i])
                h = float(cog[i]["history_size"]) if i < len(cog) else 0.0
                ca = float(cog[i]["alpha_mean"]) if i < len(cog) else float("nan")
                pa = float(per[i]["alpha_mean"]) if i < len(per) else float("nan")
                records.append({"checkpoint": checkpoint_label, "condition": condition,
                                "episode_id": eid, "timestep": ts, "action_loss": losses,
                                "offline_success_proxy": int(losses <= ns.loss_success_threshold),
                                "history_size": h, "cog_alpha": ca, "per_alpha": pa,
                                "occluded": int(bool(batch["occlusion_flags"][i]))})
            sample_count += b
            if sample_count >= len(dataset):
                break
    return records


def summarize(records, threshold):
    out = {}
    for label in sorted({r["checkpoint"] for r in records}):
        out[label] = {}
        for condition in ("normal", "occlusion"):
            rows = [r for r in records if r["checkpoint"] == label and r["condition"] == condition]
            if not rows: continue
            out[label][condition] = {
                "samples": len(rows),
                "action_loss": float(np.mean([r["action_loss"] for r in rows])),
                "offline_success_proxy": float(np.mean([r["offline_success_proxy"] for r in rows])),
                "history_size": float(np.mean([r["history_size"] for r in rows])),
                "cog_alpha": float(np.mean([r["cog_alpha"] for r in rows])),
                "per_alpha": float(np.mean([r["per_alpha"] for r in rows])),
            }
        n = { (r["episode_id"], r["timestep"]): r["action_loss"] for r in records if r["checkpoint"] == label and r["condition"] == "normal" }
        o = { (r["episode_id"], r["timestep"]): r["action_loss"] for r in records if r["checkpoint"] == label and r["condition"] == "occlusion" }
        paired = [o[k] - n[k] for k in n.keys() & o.keys()]
        out[label]["paired_occlusion_recovery"] = {
            "paired_samples": len(paired),
            "occlusion_loss_delta": float(np.mean(paired)) if paired else None,
            "occlusion_loss_delta_median": float(np.median(paired)) if paired else None,
        }
    return out


def main():
    ns = args(); seed_all(ns.seed); ns.output_root.mkdir(parents=True, exist_ok=True)
    checkpoints = dict(item.split("=", 1) for item in ns.checkpoint)
    all_records = []
    for label, path in checkpoints.items():
        model = load_checkpoint(Path(path))
        for condition in ("normal", "occlusion"):
            # Pair conditions with the same manifest order and diffusion noise.
            seed_all(ns.seed)
            all_records.extend(run_condition(model, label, condition, ns, None))
        del model; torch.cuda.empty_cache()
    csv_path = ns.output_root / "per_sample.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_records[0].keys()); writer.writeheader(); writer.writerows(all_records)
    summary = {"loss_success_threshold": ns.loss_success_threshold,
               "metric_note": "offline_success_proxy is action-loss threshold, not simulator success",
               "results": summarize(all_records, ns.loss_success_threshold)}
    (ns.output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
