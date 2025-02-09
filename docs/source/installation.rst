============
Installation
============

Express Installation
--------------------

It is recommended to use a `virtual environment <https://docs.python.org/3/library/venv.html>`_, so as not to install
anything in the global environment. If you do not wish to use a virtual environment, simply skip step 4, but first
take a look at the :ref:`Uninstalling <uninstalling>` section

.. admonition:: Quoting the official Python's documentation:

    A virtual environment is created on top of an existing Python installation, and may optionally be isolated from the 
    packages in the base environment, so only those explicitly installed in the virtual environment are available.

    When used from within a virtual environment, common installation tools such as pip will install Python packages into a 
    virtual environment without needing to be told to do so explicitly.

    A virtual environment is (amongst other things):

        * Used to contain a specific Python interpreter and software libraries and binaries which are needed to support a project 
          (library or application). These are by default isolated from software in other virtual environments and Python 
          interpreters and libraries installed in the operating system.

        * Contained in a directory, conventionally named .venv or venv in the project directory.

        * Considered as disposable - it should be simple to delete and recreate it from scratch. You don't place any project code 
          in the environment.

        * Not considered as movable or copyable - you just recreate the same environment in the target location.

In other words, a *virtual environment* is a directory on your working folder,
in which a copy of python itself and all needed packages are installed, without affecting
the global environment. To uninstall it, you simply delete the folder.

#. First, navigate to the folder you want to install.

#. Clone the depository (or simply download it from GitHub):

    .. code-block:: 

        git clone --depth 1 https://github.com/PlasmaPhy/gcmotion.git

    Your directory should look like this:

    .. code-block::

        .
        └── gcmotion
            ├── docs
            ├── examples
            ├── gcmotion
            ├── LICENSE.txt
            ├── pyproject.toml
            ├── README.md
            ├── requirements.txt
            └── tests
        
    If you want the whole git repository history, remove the :code:`--depth 1` option.

#. Navigate to the :code:`gcmotion` directory (the one containing the :code:`pyproject.toml` file):

    .. code-block::
        
        cd gcmotion

#. Install the virtual environment and activate it:

    .. code-block::

        python -m venv .venv

        source ./.venv/bin/activate.<shell>     # Linux
        ./.venv/bin/activate                    # Windows

If your prompt now shows a `(.venv)` then the virtual environment is installed. Just like that!

#. Now you can install gcmotion by using pip:

    .. code-block::

        (.venv): pip install .

And you are done!

It is recommended to make a new folder inside :code:`gcmotion/` called :code:`usr/` and keep your code there.

Installing package as `editable`
--------------------------------

If you want to make changes to the package, then in step 5 use the following command:

.. code-block::

    pip install -e .

This way the package will always be up to date. If using a Notebook, a kerner restart is required to reload the changes.


Uninstalling
------------

If you use a virtual environment, deletion is as simple as deleting the parent directory, as everything
is installed locally and nothing outside of it was affected. 

To uninstall gcmotion from the global environment, some caution is required

.. code-block::

    pip uninstall gcmotion


If you want to remove the package's dependencies as well then you want to remove it with :code:`pip-autoremove`:

.. _uninstalling:

.. warning::
    
    This action might delete dependencies such as numpy, scipy and matplotlib even if they were installed before
    gcmotion, depending on the compatibility of the versions. Use a virtual environment and save you the trouble!

.. code-block::

    pip install pip-autoremove
    pip-autoremove gcmotion -y
