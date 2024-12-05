from functools import wraps
from typing import Any, Callable, Iterable

from torch.profiler import ProfilerActivity, _ExperimentalConfig, profile, record_function
from torch.profiler.profiler import ProfilerAction


def profile_function(function: Callable) -> Callable:
    """Function decorator for profiling the function with torch

    Parameters
    ----------
    function : Callable
        callable function to profile

    Returns
    -------
    Callable
        wrapped function
    """

    @wraps(function)
    def wrapper(*args, **kwargs):
        with record_function(f"GROW::{function.__qualname__}"):
            return function(*args, **kwargs)

    return wrapper


class CustomProfile(profile):
    def __init__(
        self,
        *,
        active: bool = True,
        activities: Iterable[ProfilerActivity] | None = [
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        schedule: Callable[[int], ProfilerAction] | None = None,
        on_trace_ready: Callable[..., Any] | None = None,
        experimental_config: _ExperimentalConfig | None = None,
    ):
        super().__init__(
            activities=activities,
            schedule=schedule,
            on_trace_ready=on_trace_ready,
            record_shapes=False,
            profile_memory=True,
            with_stack=True,
            with_flops=False,
            with_modules=True,
            experimental_config=experimental_config,
            use_cuda=False,
        )
        self.active = active

    def start(self):
        if self.active:
            return super().start()
        else:
            return None

    def stop(self):
        if self.active:
            return super().stop()
        else:
            return None

    def step(self):
        if self.active:
            return super().step()
        else:
            return None


# with profile(
#     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#     profile_memory=True,
#     record_shapes=True,
#     with_stack=True,
#     experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
# ) as prof:
