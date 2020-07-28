# EIT-MPhys-Project
This code includes a novel Electrode Selection Algorithm (ESA) for use in Electrical Impedance Tomography (EIT) on 2-dimensional samples. It also features an implementation of a solver of the EIT forward problem that considers the Complete Electrode Model. The model uses the GREIT implementation of the PyEIT package (https://github.com/liubenyuan/pyEIT) for EIT computations, as well as some other parts of the PyEIT software. Our team does not have any claims for these parts of the code.

Note that the `mesh` and `eit` folders contain copies of the corresponding files found in the PyEIT package (https://github.com/liubenyuan/pyEIT). The exceptions are the `fem.py` and the other python files starting with `fem-` in the `eit` folder. They are new implementations of Finite Electrode Method solvers written exclusively by our team - Ivo Mihov and Vasil Avramov. 

## Required Packages

|  Packages        | Notes                               |
| :--------------- | :---------------------------------- |
|  NumPy           | tested with numpy 1.16.5            |
|  PyEIT *         | tested with pyeit 0.0.1             |
|  CuPy            | tested with cupy 6.0.0              |
|  Scikit-Learn    | tested with scikit-learn 0.21.3     |
|  Tensorflow 2    | tested with tensorflow-gpu 2.2.0    |
|  h5py            | tested with h5py 2.10.0             |
|  SciPy           | tested with scipy 1.4.1             |
|  Scikit-image    | tested with scikit-image 0.15.0     |
|  Matplotlib      | tested with matplotlib 3.1.1        |
|  Scikit-optimize | tested with scikit-optimize 0.7.4   |
 * PyEIT package was modified in this project, so you may need to substitute the package files with the ones included in this project to use this software.
