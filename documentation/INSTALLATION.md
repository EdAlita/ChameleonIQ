# System requirements

## Operating systems
Nema Analysis Tool has been tested on Linux, Windows and MacOS latest! It should work out of the box!

# Installation instructions
We strongly recommend that you install Nema Analysis Tool in virtual enviroment!

Use a recent version of Python! 3.11 or newer is guaranteed to work!

1) Install Nema Analysis Tool depending on your use case:
    1) For use as **out-of-the-box**:

        ```pip install ChameleonIQ```

    2) For integrative **framework** (this will create a copy of the Nema Analysis Tool code on your computer so that you can modify it as needed):

        ```bash
        git clone https://github.com/EdAlita/nema_analysis_tool.git
        cd nema_analysis_tool
        pip install -e .
        ```

2) nema_analysis_tool needs a configuration file in order to run it. Please see the section Configure file [here](USAGE.md)

Installing nema_analysis_tool will add several new commands to your terminal. These commands are used to run the entire analysis. You can execute them from any location on your system. All nema_analysis_tool commands have the prefix `chameleoniq` for easy identification.

Note that these commands simply execute python scripts. If you installed nema_analysis_tool in a virtual enviroment, this eviroment must be activated when executing the commands. You can see what scripts/functions are executed by checking the project.scripts in the [pyproject.toml](../pyproject.toml) file.

All nema_analysis_tool commands have a `-h` option which gives information on how to use them.
