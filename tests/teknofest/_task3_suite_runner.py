"""Eski Task 3 test girişlerini güncel üretim-kodu senaryolarına bağlar."""

import importlib.util
import unittest
from pathlib import Path


def run_selected(name_predicate):
    suite_path = Path(__file__).with_name("test_task3_real_sensor_flow.py")
    spec = importlib.util.spec_from_file_location("task3_real_sensor_flow", suite_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    case = module.Task3RealSensorFlowTests
    names = [name for name in unittest.defaultTestLoader.getTestCaseNames(case)
             if name_predicate(name)]
    suite = unittest.TestSuite(case(name) for name in names)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
