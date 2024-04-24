#trueOnlineTdLambdaAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math
import numpy as np
from argparse import Action

class XLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None

        # Implementing epsilon greedy here
        if legalActions:
            if util.flipCoin(self.epsilon):
                action = random.choice(legalActions)
            else:
                action = self.computeBestAction(state)

        return action

class PacmanXAgent(XLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, lmbda=0.8, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['lmbda'] = lmbda
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        XLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent (or XLearningAgent) and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = XLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class TrueOnlineTdLambdaAgent(PacmanXAgent):
    """
       Implementation of the TrueOnlineTdLambdaAgent

    """

    def getAgentName(self):
        """
            This function basically returns the name of the agent
        """
        return "TOTD"

    def __init__(self, **args):
        PacmanXAgent.__init__(self, **args)
        # self.featExtractor = util.lookup(extractor, globals())()
        self.featExtractor = SimpleExtractor3()
        self.weights = util.Counter()
        self.currentFeatureVector = util.Counter()
        self.vOld = 0
        self.z = util.Counter()

    def initWeights(self, feature_dict):
        """
            This function is used to initialize the weight vector
        """

        for key in feature_dict:
            self.weights[key] = 0

    def initTraces(self, feature_dict):
        """
            This function is used to initialize the trace vector
        """

        for key in feature_dict:
            self.z[key] = 0

    def getWeights(self):
        return self.weights

    def computeBestAction(self, state):
        """
            Essentially used for policy improvement step, where we compute the best action possible from a state
        """
        
        legalActionList = self.getLegalActions(state)
        
        # if self.isInTesting():
        #     print(legalActionList)
        
        if not legalActionList:
            return None
        
        # actionValDict = {}
            
        
        max_val = - float('inf')
        max_action = None
        for action in legalActionList:
            val = self.getStateActionValue(state, action)
            # if self.isInTesting():
            #     actionValDict[action] = val
            if val > max_val:
                max_val = val
                max_action = action
        # if self.isInTesting():
        #     print(actionValDict)
        return max_action

    def getStateActionValue(self, state, action):
        """
          Should return V(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        
        val=0
        feature_dict = self.featExtractor.getFeatures(state, action)
        for key in feature_dict.keys():
            val += self.weights.get(key, 0) * feature_dict[key]
        return val

    def getStateValue(self, feature_dict):
        """
          Should return V(state) = w * featureVector
          where * is the dotProduct operator
        """

        val = 0
        if len(feature_dict.keys()) ==0:
            return val
         
        for key in feature_dict.keys():
            val += self.weights.get(key, 0) * feature_dict[key]             
        return val
    
    def calcualteDelta(self, reward, futureStateValue, currentStateValue):
        """
            This function is used to calculate the delta parameter value
        """
        
        delta = reward + (self.discount * futureStateValue) - currentStateValue
        return delta

    def updateTraces(self, currentFeatureVector):
        """
            This function holds code to update the trace vector
        """
        
        const1 = self.discount * self.lmbda
        const2 = self.alpha * self.discount * self.lmbda
        
        for key in self.z.keys():
            self.z[key] = (const1 * self.z[key])

        for key in currentFeatureVector.keys():
            self.z[key] += (1 - (const2 * self.z.get(key, 0) * currentFeatureVector[key])) * currentFeatureVector[key]

    def startEpisode(self):
        """
            Overriding the start episode function to set vOld, z and the currentfeatureVector to 0
        """

        ReinforcementAgent.startEpisode(self)        
        self.vOld = 0
        self.z = util.Counter()
        self.currentFeatureVector = util.Counter()


    def updateWeightValues(self, delta, vVal, vOldVal, currentFeatureVector):
        """
            This function is used to update the weight vector
        """

        constantTermOne = self.alpha * (delta + vVal - vOldVal)
        constantTermTwo = self.alpha * (vVal - vOldVal)

        for key in currentFeatureVector.keys():
            self.weights[key] = self.weights.get(key, 0) + (constantTermOne * self.z.get(key, 0)) - (constantTermTwo * currentFeatureVector.get(key, 0))  


    def update(self, state, action, nextState, reward):
        """
           This function is used to hold the code for the update step
        """

        if len(self.currentFeatureVector.keys()) == 0:
            self.currentFeatureVector = self.featExtractor.getFeatures(state, action)

        if len(self.weights.keys()) == 0:
            self.initWeights(self.currentFeatureVector)
        if len(self.z.keys()) == 0:
            self.initTraces(self.currentFeatureVector)
        
        if self.computeBestAction(nextState):
            futureFeatureVector = self.featExtractor.getFeatures(nextState, self.computeBestAction(nextState))
        else:
            futureFeatureVector = util.Counter()

        vVal = self.getStateValue(self.currentFeatureVector)
        vPrimeVal = self.getStateValue(futureFeatureVector)
        delta = self.calcualteDelta(reward, vPrimeVal, vVal)
        self.updateTraces(self.currentFeatureVector)
        self.updateWeightValues(delta, vVal, self.vOld, self.currentFeatureVector)
        
        self.vOld = vPrimeVal
        self.currentFeatureVector = futureFeatureVector
        
        # self.reflection(futureFeatureVector)

    def reflection(self, futureFetaureVector):
        """
            This function is just used to print out relevant parameters for debugging
        """
        
        print("Trace: {}".format(self.z))
        print("Weights: {}".format(self.weights))
        print("Future feature vector: {}".format(futureFetaureVector))

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        # self.reflection()
        PacmanXAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            # self.reflection()
            print("\n\nCurrent weights:\n\n")
            print("Weights: {}\n\n".format(self.weights))
