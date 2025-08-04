# Relative Kinematics in Anchorless Environment

Thus is a Python library for reproducing the work published in [Estimation of Relative Kinematic Parameters in Anchorless Environments]([https://link-url-here.org](https://ieeexplore.ieee.org/abstract/document/10956134)).

## Description
There are 4 main fiolders:
* **./main:** contains the code for the simulations in the paper.
* **./output:** contains simulation output files used to generate the plots as shown in the paper. Format is **.npz**.
* **./plot:** contains the code for exactly recreating the plots in the paper.
* **./util:** contains additional code for simu8lations. Includes code for Procrustes error and other miscellaneous functions.

Key points to consider:
* The paths specified in the code are specified relative to the location of the code. Consider changing the path appropriately.
* The codes in /main folder can be run again with different kinematic parameters. Be sure to change the save folder path if necessary.


## Support and questions to the community

Ask questions using the issues section.

## Supported Platforms:

[<img src="https://www.python.org/static/community_logos/python-logo-generic.svg" height=40px>](https://www.python.org/)
[<img src="https://upload.wikimedia.org/wikipedia/commons/5/5f/Windows_logo_-_2012.svg" height=40px>](http://www.microsoft.com/en-gb/windows)
[<img src="https://upload.wikimedia.org/wikipedia/commons/8/8e/OS_X-Logo.svg" height=40px>](http://www.apple.com/osx/)
[<img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" height=40px>](https://en.wikipedia.org/wiki/List_of_Linux_distributions)

Python 3.5 and higher

## Citation

    @Misc{Mishra2023,
      author =   {{A. Mishra and R. T. Rajan}},
      title =    {{Estimation of Relative Kinematic Parameters of an Anchorless Network}},
      howpublished = {\url{[Github](https://github.com/asil-lab/AM_J1_relative_kinematics)}},
      year = {2023}
    }

## Getting started:

The code is written in Python.

### Python packages:

Packages to be installed:

    python3-dev
    build-essential   
    scipy
    numpy
    cvxpy

### Running simulations:

Constant velocity case and comparison with the State-of-the-Art:

    python3 $DIR$/main/comp_vel.py

Constant acceleration case and comparison with the State-of-the-Art::

    python3 $DIR$/main/cnst_acc.py

Constant acceleration case and effect of Signal-to-Noise (SNR) ratio:

    python3 $DIR$/main/cnst_acc_snr.py

Appendix plots:

    python3 $DIR$/plots/distance_noise.py
    python3 $DIR$/plots/lle_solvability.py

## Funding Acknowledgements

* This work is partially funded by the European Leadership Joint Undertaking (ECSEL JU), under grant agreement No 876019, the ADACORSA project - ”Airborne Data Collection on Resilient System Architectures.”
