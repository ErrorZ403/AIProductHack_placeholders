import os


class ImproperlyConfiguredError(Exception):
    def __init__(self, variable_name: str, *args: tuple, **kwargs: dict) -> None:
        self.variable_name = variable_name
        self.message = f'Set the {variable_name} environment variable.'
        super().__init__(self.message, *args, **kwargs)


def get_env_variable(var_name: str, cast_to: type = str) -> str:
    try:
        return cast_to(os.environ[var_name])
    except KeyError:
        raise ImproperlyConfiguredError(var_name) from None
    except ValueError:
        raise ValueError('Bad environment variable casting.') from None
