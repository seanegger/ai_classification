"""
Author: Sean Egger, Alec Rulev
Class: CSI-480-01
Assignment: Supervised Learning Pacman Programming Assignment
Date Assigned: Tuesday
Due Date: Monday 11:59
 
Description:
A pacman ai program
 
Certification of Authenticity: 
I certify that this is entirely my own work, except where I have given 
fully-documented references to the work of others. I understand the definition 
and consequences of plagiarism and acknowledge that the assessor of this 
assignment may, for the purpose of assessing this assignment:
- Reproduce this assignment and provide a copy to another member of academic
- staff; and/or Communicate a copy of this assignment to a plagiarism checking
- service (which may then retain a copy of this assignment on its database for
- the purpose of future plagiarism checking)
perceptron_numpy.py

Optimized perceptron implementation using numpy

Champlain College CSI-480, Fall 2017
The following code was adapted by Joshua Auerbach (jauerbach@champlain.edu)
from the UC Berkeley Pacman Projects (see license and attribution below).

----------------------
Licensing Information:  You are free to use or extend these projects for
educational purposes provided that (1) you do not distribute or publish
solutions, (2) you retain this notice, and (3) you provide clear
attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

Attribution Information: The Pacman AI projects were developed at UC Berkeley.
The core projects and autograders were primarily created by John DeNero
(denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
Student side autograding was added by Brad Miller, Nick Hay, and
Pieter Abbeel (pabbeel@cs.berkeley.edu).
"""


import util
PRINT = True
import numpy as np

class OptimizedPerceptronClassifier:
    """
    Optimized Perceptron classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    """
    def __init__( self, legal_labels, max_iterations):
        self.legal_labels = legal_labels
        self.type = "perceptron_numpy"
        self.max_iterations = max_iterations
        # note we do not yet have the information to initialize the weight
        # matrix, since it depends on the dimensionality of the feature vector
        self.weights = None
        self.features = None

    def train( self, training_data, training_labels, validation_data, validation_labels ):
        """
        The training loop for the perceptron passes through the training data several
        times and updates the weight vector for each label based on classification errors.
        See the assignment description for details.

        The data will still come in with each data point as a counter from 
        features to values for those features (and thus represents a vector a values),
        so you should first convert this to a numpy array using the
            counter.values_as_numpy_array
        method
        """

        # now we can initialize the weights
        if self.weights is None:
            self.features = list(training_data[0].keys()) # could be useful later
            self.weights = np.zeros((len(self.features), len(self.legal_labels)))

        "*** YOUR CODE HERE ***" 
        util.raise_not_defined()       
        
                    
    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...the data list will first be converted
        to a numpy array then you must return the prediction for each entry.
        """
        if self.weights is None:
            raise Exception("the perceptron must be trained before data can be classified")
        
        # convert to numpy matrix
        data_matrix = np.asarray([datum.values_as_numpy_array() for datum in data])
        
        "*** YOUR CODE HERE ***"
        util.raise_not_defined()

    def find_high_weight_features(self, label, num=100):
        """
        Returns a list of num (default 100) features with the greatest weight 
        for some label
        
        Hint: self.features stores the list of features names.  Here you will 
        have to find which rows contain the largest values in the column of 
        self.weights corresponding to the given label, and then return the 
        feature names for those rows
        
        Hint: to get the keys of a dictionary sorted by their value you can do
			sorted([key for key in dictionary.keys()], key=lambda k: dictionary[k])
		You can also set some other function or lambda expression as the sort key
        """
        
        "*** YOUR CODE HERE ***"
        util.raise_not_defined()
