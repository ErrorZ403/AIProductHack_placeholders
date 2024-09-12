import os


class ImproperlyConfigured(Exception):
    def __init__(self, variable_name: str, *args, **kwargs):
        self.variable_name = variable_name
        self.message = f"Set the {variable_name} environment variable."
        super().__init__(self.message, *args, **kwargs)


def get_env_variable(var_name: str, cast_to=str) -> str:
    try:
        return cast_to(os.environ[var_name])
    except KeyError:
        raise ImproperlyConfigured(var_name)
    except ValueError:
        raise ValueError("Bad environment variable casting.")
