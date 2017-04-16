shuffled-stats
===========
A python library for performing inference on datasets with shuffled / unordered labels. 

This library includes functions for generating datasets and performing linear regression on datasets whose labels (the "y") are shuffled with respect to the input features (the $x$). In other words, this library can be used to perform linear regression when you don't know which measurement comes from which data point.

Applications include: experiments done on an entire population of particles at once (flow cytometry), datasets shuffled to protect privacy (medical records), measurements where the ordering is unclear (signaling with identical tokens)

Examples (no noise)
-------
Let's start with some simple examples. Let's construct some 2-dimensional input data, and corresponding 

.. code-block:: python

	import numpy as np, shuffled_stats

	x = np.random.normal(1, 1, (100,2))
	y = x[:,0]*3 - 7*x[:,1]

	np.shuffle(y) #in-place shuffling of the labels

	shuffled_stats.linregress(x,y) #performs shuffled linear regression
	>>> array([3., -7.])

	#weights are exactly recovered


Clearly, the pattern here is that 
