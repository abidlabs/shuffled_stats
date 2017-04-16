shuffled-stats
===================
A python library for performing inference on datasets with shuffled / unordered labels. 

This library includes functions for generating datasets and performing linear regression on datasets whose labels (the "y") are shuffled with respect to the input features (the "x"). In other words, this library can be used to perform linear regression when you don't know which measurement comes from which data point.

Applications include: experiments done on an entire population of particles at once (flow cytometry), datasets shuffled to protect privacy (medical records), measurements where the ordering is unclear (signaling with identical tokens)

Installation
--------------------

.. code-block:: 

	$ pip install shuffled_stats


Examples (without noise)
-------------------------------
Let's start with some simple examples. Let's construct some 2-dimensional input data, and corresponding labels.

.. code-block:: python

	import numpy as np, shuffled_stats

	x = np.random.normal(1, 1, (100,2)) #input features
	y = 3*x[:,0] - 7*x[:,1] #labels

	np.shuffle(y) #in-place shuffling of the labels

	shuffled_stats.linregress(x,y) #performs shuffled linear regression
	>>> array([3., -7.])


The weights, [3, -7], were recovered exactly. 

We can do another example with defined data points:

=====  =====  =======
x1      x2    y
=====  =====  =======
1      2      3
2      5      7
-1     -2     -3
5      5      10
2      10      12
=====  =====  =======

Linear regression would clearly reveal that x1 + x2 = y. Let's see what shuffled linear regression reveals with a permuted version of the labels:

.. code-block:: python

	import numpy as np, shuffled_stats

	x = np.array([[1,2],[2,5],[-1, -2],[5,5],[2,10]])
	y = np.array([-3, 10, 7, 3, 12])

	shuffled_stats.linregress(x,y) #performs shuffled linear regression
	>>> array([1., 1.])



Examples (with noise)
------------------------
d
