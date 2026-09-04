"""Phase 0 training entry point (M4).

Usage (from repo root):
    python -m train.scripts.train_phase0 --config train/configs/phase0_dev.yaml
    python -m train.scripts.train_phase0 --config ... --resume train/runs/.../ckpt_step500.pt
"""
import argparse

import torch

from train.src.config import load_train_config
from train.src.train.trainer import Trainer
from train.utils.log import log


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default=None, help="checkpoint path to resume from")
    p.add_argument("--max-steps", type=int, default=None, help="override step count ceiling for short test runs / clean interrupts")
    args = p.parse_args()

    cfg = load_train_config(args.config)
    trainer = Trainer(cfg)

    if cfg.init == "warm":
        from train.src.tools.warm_start import warm_start_from_checkpoint
        report = warm_start_from_checkpoint(trainer.model, cfg.donor_path)
        log(f"warm start: {report['donor_consumed']}/{report['donor_tensors']} donor "
            f"tensors, {len(report['new_refit_params'])} new params",
            filename=trainer.log_file, print_console=True)
    elif cfg.init == "prebuilt":
        from train.src.tools.base_ckpt import load_base_checkpoint
        load_base_checkpoint(trainer.model, cfg.prebuilt_path)
        log(f"prebuilt base checkpoint loaded: {cfg.prebuilt_path}",
            filename=trainer.log_file, print_console=True)
    else:
        log("scratch init (ablation path — no donor warm start)",
            filename=trainer.log_file, print_console=True)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    try:
        trainer.train(max_steps=args.max_steps)
    except KeyboardInterrupt:
        log("\nKeyboardInterrupt caught — saving emergency checkpoint before exit...",
            filename=trainer.log_file, print_console=True)
        if trainer.step > 0:
            trainer.save_checkpoint(name=f"ckpt_step{trainer.step}_interrupted.pt")
        log("Clean shutdown complete.", filename=trainer.log_file, print_console=True)
    except torch.OutOfMemoryError:
        # Bring-up aid: dump the full CUDA memory breakdown before dying so the
        # OOM can be attributed (static state vs activations vs fragmentation).
        if torch.cuda.is_available():
            log("OOM — torch.cuda.memory_summary():\n" + torch.cuda.memory_summary(),
                filename=trainer.log_file, print_console=True)
        raise


if __name__ == "__main__":
    main()
