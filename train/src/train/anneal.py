"""Anneal schedule driver (spec §7.2 — ablation tooling, default final-state).

Turns the KnobSchedules from TrainConfig into per-step knob values for
RefitModel.set_anneal_state. With no schedules configured (the default run)
the model holds its final topology: window at its final value, theta_progress
at 1.0.

- window: linear-in-steps from seq_len (full) down to the config's final
  window; None (full) until the schedule starts; exactly final at end_step.
- theta: linear step progress 0 -> 1; the log-space interpolation between the
  donor and final inv_freq vectors happens inside the model.
"""
from __future__ import annotations

from typing import Optional

from train.src.config import KnobSchedule, TrainConfig


def _progress(step: int, sched: KnobSchedule) -> float:
    if step <= sched.start_step:
        return 0.0
    if step >= sched.end_step:
        return 1.0
    return (step - sched.start_step) / (sched.end_step - sched.start_step)


def window_at(step: int, sched: Optional[KnobSchedule], seq_len: int, final_window: int) -> Optional[int]:
    """None = full sequence (the anneal's "from" state). No schedule = hold
    the final window (spec v2.0: training starts at final topology)."""
    if sched is None:
        return final_window
    p = _progress(step, sched)
    if p <= 0.0:
        return None
    if p >= 1.0:
        return final_window
    w = round(seq_len - (seq_len - final_window) * p)
    return max(final_window, min(seq_len, w))


def theta_progress_at(step: int, sched: Optional[KnobSchedule]) -> float:
    """No schedule = 1.0 (final bare theta)."""
    if sched is None:
        return 1.0
    return _progress(step, sched)


class AnnealDriver:
    """Computes and applies per-step anneal state for a RefitModel."""

    def __init__(self, cfg: TrainConfig, model) -> None:
        self.cfg = cfg
        self.model = model
        assert model.refit_config.swa is not None
        self.final_window = model.refit_config.swa.window

    def state_at(self, step: int) -> dict:
        return {
            "window": window_at(step, self.cfg.anneal_window, self.cfg.seq_len, self.final_window),
            "theta_progress": theta_progress_at(step, self.cfg.anneal_theta),
        }

    def apply(self, step: int) -> dict:
        state = self.state_at(step)
        self.model.set_anneal_state(**state)
        return state
