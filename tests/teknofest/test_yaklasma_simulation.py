#!/usr/bin/env python3
"""Yaklaşma senaryoları için geriye uyumlu test giriş noktası."""

from _task3_suite_runner import run_selected


if __name__ == "__main__":
    run_selected(
        lambda name: name.startswith("test_approach_")
        or name.startswith("test_complete_approach_")
        or name == "test_low_confidence_and_target_loss_are_rejected"
    )
