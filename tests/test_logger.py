import importlib
import unittest

import torch

from gromo.utils.logger import Logger


class TestLogger(unittest.TestCase):
    def setUp(self) -> None:
        self.test_key = "test"
        self.test_value = 0.0
        self.test_tensor = torch.rand((2, 3))
        self.step = 0

    def test_deactivation(self) -> None:
        logger = Logger("Test", enabled=False)
        logger.setup_tracking(online=True)
        logger.setup_tracking(online=False)
        logger.start_run()

        logger.log_parameter(self.test_key, self.test_key)
        logger.log_artifact(self.test_key)
        logger.log_metrics({self.test_key: self.test_value}, self.step)
        logger.log_metric(self.test_key, self.test_value, self.step)
        logger.log_metric_with_stats(self.test_key, self.test_tensor, self.step)
        logger.log_pytorch_model(None, self.test_key, self.test_tensor)

        logger.save_metrics({self.test_key: self.test_value})
        logger.save_metric(self.test_key, self.test_value)
        logger.log_all_metrics(self.step)
        logger.clear()

        logger.end_run()

    def test_activation(self) -> None:
        logger = Logger("Test", enabled=True)
        if importlib.util.find_spec("mlflow") is None:
            assert not logger.enabled
        else:
            assert logger.enabled


if __name__ == "__main__":
    unittest.main()
