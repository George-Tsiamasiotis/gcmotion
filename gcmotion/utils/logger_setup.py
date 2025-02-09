"""
============
Logger setup
============

Module ``logger_setup.py``

Sets up the global logger of the system. The log file is named "log.log" and
is saved in the directory that the program is run in. The file is overwritten
upon kernel restart, so using gcm interactively appends lines until restarting
the session.

Here is how to supress and re-enable the logger:

>>> from gcmotion.utils.logger_setup import logger
>>> ...
>>> logger.disable("gcmotion")
>>> function() # function that logs stuff but we want to disable
>>> logger.enable("gcmotion")
>>> ...

"""

from loguru import logger
from gcmotion.configuration.scripts_configuration import LoggerConfig


# Grab Configuration
config = LoggerConfig()

fmt_prefix = ""
if config.module_prefix:
    fmt_prefix += "|{module}|"
if config.file_prefix:
    fmt_prefix += "|{file}|"
if config.name_prefix:
    fmt_prefix += "|{name}|"
del config.file_prefix, config.module_prefix, config.name_prefix

# Specify datetime format
if config.format in ["timedelta", ""]:

    def fmt(record):
        mins = record["elapsed"].seconds // 60
        secs = record["elapsed"].seconds % 60
        msecs = int(record["elapsed"].microseconds / 1000)
        return (
            f"{fmt_prefix}"
            f"<green>T+{mins}:{secs}:{msecs:04d}</>"
            " |{level: ^7}| {message}\n"
        )

    config.format = fmt

else:
    config.format = (
        fmt_prefix + "<green>{time:HH:mm:ss:SSS}</> |{level: ^7}| {message}"
    )


# Setup logger
logger.remove()  # Removes the default one which prints on sys.stderr
logger.add(**vars(config))
logger.opt(colors=True).info(
    f"<green>======Logger added on {config.level} level.======</>"
)
