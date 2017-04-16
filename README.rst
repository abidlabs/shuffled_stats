shuffled-stats
===========
A python library for performing inference on datasets with shuffled / unordered labels. This library includes functions for generating datasets, loading datasets, and performing linear regression on datasets whose labels (:math:`y^n`'s) are shuffled with respect to the input features (the $x$'s). 

Examples (no noise)
-------
Let's start with some simple examples. Let's construct some 2-dimensional input data, and corresponding labels:


