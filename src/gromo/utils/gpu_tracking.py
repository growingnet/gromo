import warnings
from types import TracebackType
from typing import Optional, Self, Type

import numpy as np

from gromo.config.loader import load_config
from gromo.utils.logger import Logger
from gromo.utils.utils import set_from_conf


class GpuTracker:
    """Tracking GPUs for performance

    Parameters
    ----------
    gpu_index : list[int], optional
        indices of gpus to track, by default [0]
    interval : int, optional
        time interval between measuring, by default 15
    country_iso_code : str, optional
        the country of the cluster in iso code to estimate carbon emissions
    logger : Logger | None, optional
        associated logger to track metrics, by default None

    Attributes
    ----------
    tracking : bool
        Define if tracker is activated

    _tracker : Tracker
        Tracker used to monitor GPU

    gpu_metrics : dict
        Metrics returned by logger

    Example usage: Tracking power usage continuously for a code block
    .. code-block:: python
        with GpuTracker(gpu_index=[0], interval=1) as tracker:
            # Simulate long-running code (e.g., training or other tasks)
            for i in range(5):
                print(f"Running task step {i + 1}...")
                time.sleep(1)  # Simulate work

        print(tracker.gpu_metrics)

    """

    def __init__(
        self,
        gpu_index: list[int] = [0],
        interval: int = 15,
        country_iso_code: str = "FRA",
        logger: Logger | None = None,
    ) -> None:
        self._config_data, _ = load_config()
        self.gpu_index = gpu_index
        self.interval = interval
        self._logger = logger
        # self.thread = None
        self.tracking = set_from_conf(self, "carbon_evaluation", True, setter=False)
        if self.tracking:
            self.tracking = self.__import_module()

        if self.tracking:
            self._tracker = codecarbon.OfflineEmissionsTracker(
                gpu_ids=gpu_index,
                measure_power_secs=interval,
                allow_multiple_runs=True,
                country_iso_code=set_from_conf(
                    self, "country_iso_code", country_iso_code, setter=False
                ),
                save_to_file=False,
                logging_logger=None,
                log_level="error",
            )
            self.gpu_metrics = {}

    def __enter__(self) -> Self:
        if self.tracking:
            self._tracker.start()
            return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Stop tracking power usage when exiting the context."""
        if self.tracking:
            self._tracker.stop()
            self.gpu_metrics = self._tracker.final_emissions_data.__dict__
            self.gpu_metrics["total_power"] = np.sum(
                [hardware.total_power().kW * 1000 for hardware in self._tracker._hardware]
            )
            if self._logger is not None:
                for key, value in self.gpu_metrics.items():
                    if key in ["timestamp", "project_name", "run_id", "experiment_id"]:
                        continue
                    if not isinstance(value, (int, float)):
                        self._logger.log_parameter(key, str(value))
                    else:
                        self._logger.save_metric(name=f"codecarbon/{key}", value=value)

    def __import_module(self) -> bool:
        try:
            global codecarbon
            import codecarbon

            tracking = True
        except ImportError as err:
            warnings.warn(f"{err}. Energy tracking will be skipped.", ImportWarning)
            tracking = False

        return tracking
