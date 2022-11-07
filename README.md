[![Python 3.10](https://img.shields.io/badge/python-3.10.6-blue.svg)](https://www.python.org/downloads/release/python-3106/)
[![license](https://img.shields.io/badge/license-apache_2.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)

![Alt Text](README_Front_Image.gif)
## NSMA: A Memetic Procedure for Global Multi-Objective Optimization

Implementation of the NSMA algorithm proposed in

[Lapucci, M., Mansueto, P., & Schoen, F. (2022). A Memetic Procedure for Global Multi-Objective Optimization. arXiv preprint arXiv:2201.11444.](
https://doi.org/10.48550/arXiv.2201.11444)

If you have used our code for research purposes, please cite the publication mentioned above.
For the sake of simplicity, we provide the Bibtex format:

```
@misc{nsma,
  doi = {10.48550/ARXIV.2201.11444},
  url = {https://arxiv.org/abs/2201.11444},
  author = {Lapucci, Matteo and Mansueto, Pierluigi and Schoen, Fabio},
  title = {A Memetic Procedure for Global Multi-Objective Optimization},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

### Installation

In order to execute the code, you need an [Anaconda](https://www.anaconda.com/) environment. We provide YAML file in order to facilitate the installation of the latter.

##### For Windows user

Open an Anaconda Prompt in the project root folder and execute the following command.

```
conda env create -f Environment_Setups/windows_environment_setup.yml
```

##### For Linux user

Open a terminal in the project root folder and execute the following command.

```
conda env create -f Environment_Setups/linux_environment_setup.yml
```

##### For MacOSX user

Open a terminal in the project root folder and execute the following command.

```
conda env create -f Environment_Setups/macosx_environment_setup.yml
```

#### Main Packages

* ```python v3.10.6```
* ```pip v22.2.2```
* ```numpy v1.22.3```
* ```scipy v1.7.3```
* ```matplotlib``` Windows: ```v3.5.3```, Linux: ```v3.5.2```, MacOSX: ```v3.6.1```
* ```tensorflow``` Windows: ```v2.9.1```, Linux: ```v2.8.2```, MacOSX: ```v2.10.0```
* ```gurobipy v9.5.2```
* ```progressbar2 v4.2.0```

#### Gurobi Optimizer

In order to run some parts of the code, the [Gurobi](https://www.gurobi.com/) Optimizer needs to be installed and, in addition, a valid Gurobi licence is required. 
However, the employment of the Gurobi Optimizer is not mandatory to execute the code. 
Indeed, we provide alternative scripts where the HiGHS dual simplex solver implementation by [SciPy](https://scipy.org/) is used. The latter is the default choice as can be seen in ```parser_management.py```. 
The Gurobi optimizer can be only employed by activating an argument (```-g```, ```--gurobi```). 

### Usage

We refer to the code documentation for all the information. In particular, in ```parser_management.py``` you can find all the possible arguments.
Given a terminal (an Anaconda prompt, if you are a Windows user), an example of execution could be the following.

```python main.py --algorithms NSMA --problems MAN --seeds 16007 --max_time 2 --verbose --plot_pareto_front --plot_pareto_solutions -g --general_export --general_export_pareto_solutions```

The execution results are saved in the ```Execution_Outputs``` folder. In ```General_Utils/Management_Utils.py``` and ```General_Utils/Args_Utils.py```, you can find all the documentation about how the outputs are stored.

### Contact

If you have any question, feel free to contact me:

[Pierluigi Mansueto](https://webgol.dinfo.unifi.it/pierluigi-mansueto/)<br>
Global Optimization Laboratory ([GOL](https://webgol.dinfo.unifi.it/GOL))<br>
University of Florence<br>
Email: pierluigi dot mansueto at unifi dot it
