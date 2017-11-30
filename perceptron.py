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
perceptron.py

Basic perceptron implementation

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

class PerceptronClassifier:
    """
    Perceptron classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    """
    def __init__( self, legal_labels, max_iterations):
        self.legal_labels = legal_labels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legal_labels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def set_weights(self, weights):
        assert len(weights) == len(self.legal_labels);
        self.weights = weights;

    def train(self, training_data, training_labels, validation_data, validation_labels ):
        """
        The training loop for the perceptron passes through the training data several
        times and updates the weight vector for each label based on classification errors.
        See the assignment description for details.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        (and thus represents a vector a values).
        
        In this case, we will not use validation_data or validation_labels
        
        Important: Do not change this function!
        """

        self.features = list(training_data[0].keys()) # could be useful later

        for iteration in range(self.max_iterations):
            print("Starting iteration ", iteration, "...")
            for datum, label in zip(training_data, training_labels):
                self.perform_update(datum, label)

    def perform_update(self, datum, label):
        """
        Update the weights based on a single data point (datum) with the given 
        label.
        
        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        (and thus represents a vector a values).
        """
        
        "*** YOUR CODE HERE ***"
        counter = util.Counter()
        for i in self.legal_labels:
            counter[i] = self.weights[i] * datum
        max_score = counter.arg_max()
        if label != max_score:
            self.weights[label] += datum
            self.weights[max_score] -= datum

        

    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legal_labels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.arg_max())
        return guesses
        
