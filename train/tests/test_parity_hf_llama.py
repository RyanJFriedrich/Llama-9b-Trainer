"""M1 acceptance: our plain decoder bit-matches the HF reference.

Two tiers:
- Tiny random models (always run): same weights copied into both
  implementations, frozen probe batch, fp32, tight tolerance. Covers both the
  plain RoPE path (Llama-3.0 style) and the llama3-scaled path (the donor's
  actual RoPE config).
- The real donor checkpoint (skips until weights are delivered and CUDA is
  available): full 8B parity in bf16 on a frozen probe batch.

Frozen probe batch: fixed-seed random token ids, so the test is reproducible
across machines and runs.
"""
import json
from pathlib import Path

import pytest
import torch

from train.src.model.decoder import LlamaBaseConfig, LlamaBaseModel
from train.utils.log import log

DONOR_DIR = Path(__file__).parents[2] / "OriginalModel"

LLAMA3_SCALING = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}


def _probe_batch(vocab_size: int, batch: int = 2, seq_len: int = 64) -> torch.Tensor:
    g = torch.Generator().manual_seed(0xC0FFEE)
    return torch.randint(0, vocab_size, (batch, seq_len), generator=g)


def _tiny_hf_model(rope_scaling, seed: int = 1234):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    hf_config = LlamaConfig(
        vocab_size=997,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        rope_scaling=rope_scaling,
        max_position_embeddings=512,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )
    return LlamaForCausalLM(hf_config).eval()


@pytest.mark.parametrize("rope_scaling", [None, LLAMA3_SCALING], ids=["plain", "llama3-scaled"])
def test_parity_tiny_random(rope_scaling):
    hf_model = _tiny_hf_model(rope_scaling)
    config = LlamaBaseConfig.from_hf(hf_model.config)
    ours = LlamaBaseModel(config).eval()
    missing, unexpected = ours.load_state_dict(hf_model.state_dict(), strict=False)
    assert not missing and not unexpected

    input_ids = _probe_batch(config.vocab_size)
    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits
        our_logits = ours(input_ids)

    diff = (hf_logits - our_logits).abs()
    assert torch.allclose(hf_logits, our_logits, atol=1e-5, rtol=1e-5), (
        f"parity failed: max abs diff {diff.max().item():.3e}, "
        f"mean {diff.mean().item():.3e}"
    )


def _donor_ready() -> bool:
    """Fingerprint check for Llama-3.1-8B-Instruct (the drop strips
    _name_or_path). 3.1 signatures: 131072 max positions + explicit llama3
    rope_scaling. Instruct (not base): generation_config eos is the
    [128001, 128008, 128009]-style list including 128009 (<|eot_id|>),
    not the base card's plain 128001."""
    cfg_path = DONOR_DIR / "config.json"
    gen_path = DONOR_DIR / "generation_config.json"
    if not cfg_path.exists():
        return False
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("max_position_embeddings") != 131072:
        return False
    if (cfg.get("rope_scaling") or {}).get("rope_type") != "llama3":
        return False
    if gen_path.exists():
        eos = json.loads(gen_path.read_text()).get("eos_token_id")
        if 128009 not in (eos if isinstance(eos, list) else [eos]):
            return False
    return True


@pytest.mark.skipif(not _donor_ready(), reason="donor (Llama-3.1-8B-Instruct) not in OriginalModel/")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="8B parity run needs CUDA")
def test_parity_donor_8b():
    """M1 acceptance on the real donor: logits match HF on a frozen probe batch."""
    from transformers import AutoModelForCausalLM

    from train.src.tools.load_donor import load_donor

    device = "cuda"
    hf_model = AutoModelForCausalLM.from_pretrained(
        DONOR_DIR, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device).eval()
    ours = load_donor(DONOR_DIR, device=device, dtype=torch.bfloat16).eval()

    input_ids = _probe_batch(ours.config.vocab_size, batch=1, seq_len=128).to(device)
    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits
        our_logits = ours(input_ids)

    diff = (hf_logits - our_logits).abs()
    log(
        f"donor parity: max abs diff {diff.max().item():.3e}, mean {diff.mean().item():.3e} "
        f"(bf16, eager, probe 1x128)",
        print_console=True,
    )
    # Measured on the 8B checkpoint at bf16/eager: diff is exactly 0.0
    # (identical math, same kernels). Keep modest headroom for kernel or
    # transformers-version drift, well inside the spec's ~1e-4 bf16 bound
    # relative to logit magnitudes O(10).
    assert diff.max().item() < 1e-3
    assert diff.mean().item() < 1e-5
