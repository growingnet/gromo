import logging
import sys

import numpy as np
import torch


class Logger:
    def __init__(
        self,
        experiment_name: str,
        port: int = 27027,
        api: str = "mlflow",
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.api = api.lower().strip()
        self.__implemented_apis = ["mlflow"]
        assert (
            self.api in self.__implemented_apis
        ), "Choose implemented tracking API from {self.__implemented_apis}. Found {self.api}"
        self.__choose_module()

        if self.enabled:
            self.experiment_name = experiment_name
            self.default_port = port

            self.metrics: dict = {}
            logging.getLogger("mlflow").setLevel(logging.DEBUG)

    def setup_tracking(self, online: bool = False, port: int | None = None) -> None:
        """Set up remote tracking with logging server

        Parameters
        ----------
        online : bool, optional
            connect with logging server online instead of locally, by default False
        port : int | None, optional
            port number, by default None
        """
        if not self.enabled:
            return
        if port is None:
            port = self.default_port
        if self.api == "mlflow":
            self.__setup_mlflow_tracking(self.experiment_name, port, online)

    def start_run(self, **kwargs) -> None:
        """Start Logging run"""
        if not self.enabled:
            return
        if self.api == "mlflow":
            self.__start_mlflow_run(**kwargs)

    def end_run(self):
        """End logging run"""
        if not self.enabled:
            return
        if self.api == "mlflow":
            self.__end_mlflow_run()

    def log_parameter(self, key: str, value: str) -> None:
        """Log a parameter

        Parameters
        ----------
        key : str
            name of parameter
        value : str
            value of parameter
        """
        if not self.enabled:
            return
        if self.api == "mlflow":
            mlflow.log_param(key, value)

    def log_artifact(self, path: str) -> None:
        """Log an artifact

        Parameters
        ----------
        path : str
            file path of artifact
        """
        if not self.enabled:
            return
        if self.api == "mlflow":
            mlflow.log_artifact(path)

    def log_metrics(self, metrics: dict, step: int) -> None:
        """Log multiple metrics

        Parameters
        ----------
        metrics : dict
            metrics dictionary
        step : int
            index in time of metrics
        """
        if not self.enabled:
            return
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def log_metric(self, key: str, value, step: int) -> None:
        """Log single metric

        Parameters
        ----------
        key : str
            name of metric
        value : _type_
            value of metric
        step : int
            index in time of metric
        """
        if not self.enabled:
            return
        if self.api == "mlflow":
            mlflow.log_metric(key, value, step)

    def log_metric_with_stats(self, name: str, value: torch.Tensor, step: int) -> None:
        """Log several matrix metrics on mlflow
        If the tensor is one-dimentional simply log its value

        Parameters
        ----------
        name : str
            name of the matrix
        value : torch.Tensor
            value of matrix
        step : int
            index in time of metric
        """
        if not self.enabled:
            return
        if not torch.is_tensor(value):
            self.log_metric(name, value, step=step)
            return
        elif value.dim() < 1:
            self.log_metric(name, value, step=step)
            return

        self.log_metric(f"{name}/mean", torch.mean(value), step=step)
        self.log_metric(f"{name}/median", torch.median(value), step=step)
        self.log_metric(f"{name}/std", torch.std(value), step=step)
        self.log_metric(f"{name}/max", torch.max(value), step=step)
        self.log_metric(f"{name}/fr-norm", torch.linalg.norm(value), step=step)
        self.log_metric(
            f"{name}/norm",
            torch.linalg.norm(value) / np.sqrt(torch.numel(value)),
            step=step,
        )

    def log_pytorch_model(self, model, name, x) -> None:
        if not self.enabled:
            return
        if self.api == "mlflow":
            signature = mlflow.models.infer_signature(
                x.cpu().numpy(), model(x).detach().cpu().numpy()
            )
            mlflow.pytorch.log_model(model, f"{name}", signature=signature)

    def save_metrics(self, metrics: dict) -> None:
        """Save multiple growth statistics for later logging

        Parameters
        ----------
        metrics : dict
            metrics dictionary
        """
        if not self.enabled:
            return
        for key, value in metrics.items():
            self.save_metric(key, value)

    def save_metric(self, name: str, value: torch.Tensor | int | float) -> None:
        """Save growth statistics for later logging

        Parameters
        ----------
        name : str
            name of tensor
        value : torch.Tensor | int | float
            value of tensor
        """
        if not self.enabled:
            return
        self.metrics[name] = value

    def log_all_metrics(self, step: int, with_stats: bool = True) -> None:
        """Log all saved metrics

        Parameters
        ----------
        step : int
            index in time of metrics
        with_stats : bool, optional
            if there are matrices log their statistics, by default True
        """
        if not self.enabled:
            return
        for key, value in self.metrics.items():
            if with_stats:
                self.log_metric_with_stats(key, value, step)
            else:
                self.log_metric(key, value, step)

    def clear(self) -> None:
        """Clear all saved metrics"""
        if not self.enabled:
            return
        self.metrics.clear()

    def __choose_module(self) -> None:
        try:
            if self.api == "mlflow":
                global mlflow
                import mlflow
        except ImportError as err:
            print(err)
            print("Logging will be skipped")
            self.enabled = False

    def __start_mlflow_run(self, tags: dict | None = None) -> None:
        mlflow.start_run(log_system_metrics=True, tags=tags)

    def __setup_mlflow_tracking(
        self, experiment_name: str, port: int = 27027, online: bool = False
    ) -> None:
        """Set up mlflow online tracking

        Parameters
        ----------
        experiment_name : str
            name for experiment bucket on mlflow server
        port : int, optional
            port number, by default 27027
        online : bool, optional
            connect with mlflow server online instead of locally, by default False
        """
        uri = f"http://127.0.0.1:{port}"
        if online:
            mlflow.set_tracking_uri(uri=uri)
        mlflow.set_experiment(experiment_name)
        print(f"Mlflow tracking at {mlflow.get_tracking_uri()}")

    def __end_mlflow_run(self) -> None:
        mlflow.end_run()
