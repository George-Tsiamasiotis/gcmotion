"""
Logger setup
------------

Module ``logger_setup.py``

Sets up the global logger of the system. The log file is named "log.log" and
is saved in the directory that the program is run in. The file is overwritten
upon kernel restart, so using gcm interactively appends lines until restarting
the session.

Here is how to supress and re-enable the logger:

.. code-block:: python

    >>> from gcmotion.utils.logger_setup import logger
    >>> ...
    >>> logger.disable("gcmotion")
    >>> function() # function that logs stuff but we want to disable
    >>> logger.enable("gcmotion")
    >>> ...

"""

from loguru import logger


# Setup logger
logger.remove()  # Removes the default one which prints on sys.stderr


# Format template
fmt = "{time:HH:mm:ss:SSS} |{level: ^7}| {message}"

level = "DEBUG"

logger.add("log.log", format=fmt, level=level, mode="w")

logger.info(f"Logger added on {level} level.")
