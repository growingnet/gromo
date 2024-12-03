import logging

import mlflow
import numpy as np
import torch


class Logger:
    def __init__(
        self, experiment_name: str, port: int = 27027, api: str = "mlflow"
    ) -> None:
        self.experiment_name = experiment_name
        self.default_port = port
        self.api = api.lower().strip()
        self.__implemented_apis = ["mlflow"]
        assert (
            self.api in self.__implemented_apis
        ), "Choose implemented tracking API from {self.__implemented_apis}. Found {self.api}"

        self.metrics: dict = {}
        logging.getLogger("mlflow").setLevel(logging.DEBUG)

    def setup_tracking(self, online: bool = False, port: int | None = None) -> None:
        if port is None:
            port = self.default_port
        if self.api == "mlflow":
            self.__setup_mlflow_tracking(self.experiment_name, port, online)

    def start_run(self, **kwargs) -> None:
        if self.api == "mlflow":
            self.__start_mlflow_run(**kwargs)

    def end_run(self):
        if self.api == "mlflow":
            self.__end_mlflow_run()

    def log_parameter(self, key: str, value: str) -> None:
        if self.api == "mlflow":
            mlflow.log_param(key, value)

    def log_artifact(self, path) -> None:
        if self.api == "mlflow":
            mlflow.log_artifact(path)

    def log_metrics(self, metrics: dict, step: int) -> None:
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def log_metric(self, key: str, value, step: int) -> None:
        if self.api == "mlflow":
            mlflow.log_metric(key, value, step)

    def log_metric_with_stats(self, name: str, value: torch.Tensor, step: int) -> None:
        """
        Log several matrix metrics on mlflow
        If the tensor is one-dimentional simply log its value
        :param str name: name of the matrix
        :param torch.Tensor value: value of matrix
        """
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
        if self.api == "mlflow":
            signature = mlflow.models.infer_signature(
                x.cpu().numpy(), model(x).detach().cpu().numpy()
            )
            mlflow.pytorch.log_model(model, f"{name}", signature=signature)

    def save_metrics(self, metrics: dict) -> None:
        for key, value in metrics.items():
            self.save_metric(key, value)

    def save_metric(self, name: str, value: torch.Tensor | int | float) -> None:
        """
        Save growth statistics
        :param str name: name of tensor
        :param torch.Tensor value: value of tensor
        """
        self.metrics[name] = value

    def log_all_metrics(self, step: int, with_stats: bool = True) -> None:
        for key, value in self.metrics.items():
            if with_stats:
                self.log_metric_with_stats(key, value, step)
            else:
                self.log_metric(key, value, step)

    def clear(self) -> None:
        self.metrics.clear()

    def __start_mlflow_run(self, tags: dict | None = None) -> None:
        mlflow.start_run(log_system_metrics=True, tags=tags)

    def __setup_mlflow_tracking(
        self, experiment_name: str, port: int = 27027, online: bool = False
    ) -> None:
        """
        Set up mlflow online tracking
        :param str experiment_name: name for experiment bucket on mlflow server
        :param int port: port number
        :param bool online: connect with mlflow server online or locally
        """
        uri = f"http://127.0.0.1:{port}"
        if online:
            mlflow.set_tracking_uri(uri=uri)
        mlflow.set_experiment(experiment_name)
        print(f"Mlflow tracking at {mlflow.get_tracking_uri()}")

    def __end_mlflow_run(self) -> None:
        mlflow.end_run()
