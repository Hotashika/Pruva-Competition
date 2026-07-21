#!/usr/bin/env python3
"""Fiziksel çarpma senaryoları için geriye uyumlu test giriş noktası."""

from _task3_suite_runner import run_selected


if __name__ == "__main__":
    run_selected(lambda name: name.startswith("test_collision_"))
