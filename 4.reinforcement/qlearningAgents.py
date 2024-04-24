# qlearningAgents.py
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

class SemiGradientTDAgent(ReinforcementAgent):
    def __init__(self, epsilon=0.05, alpha=0.3, gamma=0.8, lambda_=0.9, numTraining=0, **args):
        args['numTraining'] = numTraining
        ReinforcementAgent.__init__(self, **args)
        self.index = 0
        self.featExtractor = SimpleExtractor3()
        self.epsilon=epsilon
        self.alpha=alpha
        self.index = 0
        self.gamma = gamma
        self.lambda_ = lambda_
        self.weights = util.Counter()  
        self.z = util.Counter() 

    def getAgentName(self):
        return "SGTD"

    def getAction(self, state):
        legalActions = self.getLegalActions(state)
        action = None
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            qValues = [self.getQValue(state, a) for a in legalActions]
            maxQ = max(qValues)
            count = qValues.count(maxQ)
            if count > 1:
                best = [i for i in range(len(legalActions)) if qValues[i] == maxQ]
                i = random.choice(best)
            else:
                i = qValues.index(maxQ)
            action = legalActions[i]
        self.lastState = state
        self.lastAction = action
        return action

    def startEpisode(self):
        ReinforcementAgent.startEpisode(self)
        self.z = util.Counter()

    def update(self, state, action, nextState, reward):
        features = self.featExtractor.getFeatures(state, action)
        qValue = self.getQValue(state, action)
        nextQValue = self.getValue(nextState)
        delta = reward + self.gamma * nextQValue - qValue
    
        for f, v in features.items():
            self.z[f] *= self.gamma * self.lambda_
            self.z[f] += v 
            self.weights[f] += self.alpha * delta * self.z[f] 

    def getValue(self, state):
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            return 0.0
        return max([self.getQValue(state, a) for a in legalActions])
    
    def getQValue(self, state, action):
        features = self.featExtractor.getFeatures(state, action)
        if features is None:
            raise ValueError("Features returned from feature extractor is None.")
        return sum(self.weights[f] * v for f, v in features.items())
    
class QLearningAgent(ReinforcementAgent):
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

        self.qValues = util.Counter() 

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.qValues[(state, action)]
        util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return 0.0
        
        return max(self.getQValue(state, action) for action in legalActions)
        util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
          return None

    # Initialize variables to store the best action and the highest Q-value
        bestAction = None
        highestQValue = float('-inf')
    
        for action in legalActions:
          qValue = self.getQValue(state, action)
        # Update the best action if the current Q-value is higher than what we've seen so far
          if qValue > highestQValue:
            highestQValue = qValue
            bestAction = action
        # If the current Q-value is equal to the highest Q-value, we randomly choose between the two actions
          elif qValue == highestQValue:
            bestAction = random.choice([bestAction, action])

        return bestAction
        util.raiseNotDefined()

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
    
        # If there are no legal actions, return None
        if not legalActions:
          return action
    
        # With probability epsilon, choose a random action
        if util.flipCoin(self.epsilon):
          action = random.choice(legalActions)
        else:
        # Otherwise, choose the best action based on the current Q-values
          action = self.computeActionFromQValues(state)
    
        return action
        util.raiseNotDefined()


    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        update=self.qValues[(state, action)] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * sample
        return update
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.75,gamma=0.80,alpha=0.2, numTraining=0, **args):
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
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        # self.featExtractor = util.lookup(extractor, globals())()
        self.featExtractor = SimpleExtractor2()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getAgentName(self):
        return "AQA"

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        features = self.featExtractor.getFeatures(state, action)
        qValue = sum(self.weights[feature] * value for feature, value in features.items())
        return qValue
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        difference = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action)
        
        # Update weights
        features = self.featExtractor.getFeatures(state, action)
        for feature, value in features.items():
            self.weights[feature] += self.alpha * difference * value
        return features
        util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            
            pass
