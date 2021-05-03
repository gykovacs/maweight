maweight
========

A Python package for multi-atlas based weight estimation for CT images, including segmentation by registration, feature extraction and model selection for regression.

About
-----

A detailed description of the implemented methodology can be found in the paper:

The package is used intensively in the case study of estimating weights of meat cuts from the CT images of rabbit in the repository: https://github.com/gykovacs/rabbit_ct_weights

If you use the package, please consider citing the paper:

.. code-block:: BibTex

    @article{Csoka2021,
        author={\'Ad\'am Cs\'oka and Gy\"orgy Kov\'acs and Vir\'ag \'Acs and Zsolt Matics and Zsolt Gerencs\'er and Zsolt Szendr\"o and \"Ors Petneh\'azy and Imre Repa and Mariann Moizs and Tam\'as Donk\'o},
        title={Multi-atlas segmentation based estimation of weights from CT scans in farm animal imaging and its applications to rabbit breeding programs},
        year={2021}
    }


Installation (Windows/Linux/Mac)
-----------

Prerequisites: elastix
**********

Make sure the elastix package (https://elastix.lumc.nl/) is installed and available in the command line by issuing

.. code-block:: bash

    > elastix

If elastix is properly installed, the following textual output should appear in the terminal:

.. code-block:: bash

    Use "elastix --help" for information about elastix-usage.


Installing the ```maweight``` package
***********

Clone the GitHub repository:

.. code-block:: bash

    > git clone git@github.com:gykovacs/maweight.git


Navigate into the root directory of the repository:

.. code-block:: bash

    > cd maweight

Install the code into the active Python environment

.. code-block:: bash

    > pip install .


Usage examples
----------

Segmentation by elastic registration
********

The main functionality of the package is registering image A to image B by elastic registration and then transforming a set of images C, D, ... to image B by the same transformation field. This functionality is implemented in the ```register_and_transform``` function:

.. code-block:: Python

    from maweight import register_and_transform

    A # path, ndarray or Nifti1Image - the atlas image
    B # path, ndarray or Nifti1Image - the unseen image
    [C, D] # paths, ndarrays or Nifti1Image objects - the atlas annotations for A, to be transformed to B
    [C_transformed_path, D_transformed_path] # paths of the output images

    register_and_transform(A, B, [C, D], [C_transformed_path, D_transformed_path])

Feature extraction
******

Given an image B and a set of atlases registered to it [C, D, ...], with corresponding labels [Clabel, Dlabel, ...] (for the labeling of features), feature extraction with bin boundaries [b0, b1, ...] can be executed in terms of the ```extract_features_3d``` function:

.. code-block:: Python

    from maweight import extract_features_3d

    B # path, ndarray or Nifti1Image - a base image to extract features from
    registered_atlases # list of paths, ndarrays or Nivti1Image objects
    labels # list of labels of the atlases (used to label the features)
    bins= [0, 20, 40, 60, 80, 100] # bin boundaries for histogram feature extraction

    features= extract_features_3d(B, registered_atlases, labels, bins)

Model selection
*******

Given a dataset of features extracted from the ensemble of segmentations, one can carry out regression model fitting by the ```model_selection``` function:

.. code-block:: Python

    from maweight import model_selection

    features # pandas DataFrame of features
    targets # pandas Series of corresponding weights

    results= model_selection(features, targets)


By default, the model selection runs simulated annealing based feature ssubset and regressor parameter selection for kNN, linear, lasso, ridge and PLS regression and returns the summary of results in a pandas DataFrame.
