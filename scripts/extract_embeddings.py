"""Extract frozen embeddings from a trained HolterSSL model (with AMP)."""
import os, sys, argparse, yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.holter_ssl import HolterSSL
from src.data.dataset import HolterPretrainDataset, collate_pretrain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and not args.no_amp
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_cfg = cfg["model"]

    model = HolterSSL(model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Support both raw checkpoints and DeepSpeed checkpoints
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "module" in ckpt:
        model.load_state_dict(ckpt["module"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    dataset = HolterPretrainDataset(
        cfg["data"]["processed_dir"], mode="ordered",
        max_windows=cfg["training"].get("max_windows", 288),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=collate_pretrain, num_workers=0)

    all_day_embeds = {}
    all_window_embeds = {}
    all_quality_masks = {}

    print(f"Extracting embeddings for {len(dataset)} patients...")
    print(f"AMP: {'enabled (' + str(amp_dtype) + ')' if use_amp else 'disabled'}")

    with torch.no_grad():
        for batch in loader:
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                outputs = model(batch_dev)

            pid = batch["patient_id"][0]
            all_day_embeds[pid] = outputs["day_embed"][0].float().cpu().numpy()
            all_window_embeds[pid] = outputs["window_embeds"][0].float().cpu().numpy()
            all_quality_masks[pid] = batch["quality_mask"][0].numpy()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(args.output,
             day_embeddings=np.array(list(all_day_embeds.values())),
             patient_ids=np.array(list(all_day_embeds.keys())),
             window_embeddings=all_window_embeds,
             quality_masks=all_quality_masks)
    print(f"Saved embeddings to {args.output}")


if __name__ == "__main__":
    main()
