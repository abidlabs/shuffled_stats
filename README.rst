shuffled-stats
===================
A python library for performing inference on datasets with shuffled / unordered labels. 

This library includes functions for generating datasets and performing linear regression on datasets whose labels (the "y") are shuffled with respect to the input features (the "x"). In other words, you should use this library to perform linear regression when you don't know which measurement comes from which data point.

Applications include: experiments done on an entire population of particles at once (`flow cytometry <https://en.wikipedia.org/wiki/Flow_cytometry>`_), datasets shuffled to protect privacy (`medical records <https://experts.illinois.edu/en/publications/protection-of-health-information-in-data-mining>`_), measurements where the ordering is unclear (`signaling with identical tokens <http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6620545>`_)

Installation
--------------------

.. code-block:: 

	$ pip install shuffled_stats


Examples (without noise)
-------------------------------
Let's start with some simple examples. We construct some random 2-dimensional input data, and the corresponding labels. We then apply shuffled regression using the :code:`shuffled_stats.linregress` function.

.. code-block:: python

	import numpy as np, shuffled_stats

	x = np.random.normal(1, 1, (100,2)) #input features
	y = 3*x[:,0] - 7*x[:,1] #labels

	np.shuffle(y) #in-place shuffling of the labels

	shuffled_stats.linregress(x,y) #performs shuffled linear regression
	>>> array([3., -7.])


The original weights, [3, -7], are recovered exactly. 

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

Linear regression would clearly reveal that 1*x1 + 1*x2 = y. Let's see what shuffled linear regression reveals with a permuted version of the labels:

.. code-block:: python

	import numpy as np, shuffled_stats

	x = np.array([[1,2],[2,5],[-1, -2],[5,5],[2,10]])
	y = np.array([-3, 10, 7, 3, 12])

	shuffled_stats.linregress(x,y) #performs shuffled linear regression
	>>> array([1., 1.])


Again, the weights are recovered exactly.

Examples (with noise)
------------------------

.. code-block:: python
	
	np.random.seed(1) #for reproducibility
	x = np.random.normal(1, 1, (100,3)) #input features
	x[:,0] = 1 #making a bias/intercept column 
	y = 4 + 2*x[:,1] - 3*x[:,2] #labels

	y = y + np.random.normal(0, .3, (100)) #adding Gaussian noise

	w = shuffled_stats.linregress(x,y)
	np.round(w,2)
	>>> array([3.80, 2.09, -2.91])

We see that the recovered weights approximate the original weights (4, 2, -3), including the bias term.

The library includes a function, :code:`shuffled_stats.generate_dataset`  to quickly generating datasets for testing. Here's an example:

.. code-block:: python
	
	np.random.seed(1) #for reproducibility

	x, y, w0 = shuffled_stats.generate_dataset(n=100, dim=3, bias=True, noise=0.3, mean=2)
	w = shuffled_stats.linregress(x,y)

	print(np.round(w0,2))
	>>> array([2.07, -1.47, -0.83])	
	print(np.round(w,2))
	>>> array([1.79, 1.55, -0.63])

The weights are approximately recovered. Can we improve performance by running three separate "trials" of this experiment, each consisting of 100 unordered labels (within each trial, the ordering of the labels is unknown, but labels within a trial must correspond to data points from that trial)? We can test this easily with our library:

.. code-block:: python
	
	#fix weights to the same values as before
	x, y, w0, groups = shuffled_stats.generate_dataset(n=300, dim=3, weights=[2.07, -1.47, -0.83], bias=True, noise=0.3, mean=2, n_groups=3)
	w = shuffled_stats.linregress(x,y, groups=groups)

	print(np.round(w,2))
	>>> array([2.09, -1.49, -0.82])

The weights are a lot closer this time!

