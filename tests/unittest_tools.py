import functools


def unittest_parametrize(param_list):
    """Decorates a test case to run it as a set of parametrized args."""

    def decorator(f):
        @functools.wraps(f)
        def wrapped(self):
            for param in param_list:
                with self.subTest(**param):
                    f(self, **param)

        return wrapped

    return decorator
