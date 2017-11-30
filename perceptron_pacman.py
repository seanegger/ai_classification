"""perceptron_pacman.py

Perceptron implementation for apprenticeship learning

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
from perceptron import PerceptronClassifier
from pacman import GameState

PRINT = True


class PerceptronClassifierPacman(PerceptronClassifier):
    def __init__(self, legal_labels, max_iterations):
        PerceptronClassifier.__init__(self, legal_labels, max_iterations)
        self.weights = util.Counter()

    def classify(self, data ):
        """
        Data contains a list of (datum, legal moves)
        
        Datum is a Counter representing the features of each GameState.
        legal_moves is a list of legal moves for that GameState.
        """
        guesses = []
        for datum, legal_moves in data:
            vectors = util.Counter()
            for l in legal_moves:
                vectors[l] = self.weights * datum[l]
            guesses.append(vectors.arg_max())
        return guesses


    def train( self, training_data, training_labels, validation_data, validation_labels ):
        self.features = list(training_data[0][0]['Stop'].keys()) # could be useful later
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.

        for iteration in range(self.max_iterations):
            print("Starting iteration ", iteration, "...")
            for (datum, legal_moves), label in zip(training_data, training_labels):
                "*** YOUR CODE HERE ***"
                print (datum)
                counter = util.Counter()
                for direction in legal_moves:
                    if datum[direction] != 0:
                        counter[direction] = self.weights * datum[direction]
                max_score = counter.arg_max()
                if label != max_score:
                    self.weights += datum[label]
                    self.weights -= datum[counter.arg_max()]
