from multiagents.util import manhattanDistance
import random
import multiagents.util as util
from multiagents.game import Agent


class ReflexAgent(Agent):

    def getAction(self, gameState):
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        evf_result = successorGameState.getScore()
        food_distance = [manhattanDistance(newPos, i) for i in newFood.asList()]

        if action == "Stop":
            evf_result -= 150

        for i in range(len(newGhostStates)):
            if (newGhostStates[i].getPosition() == newPos and (newScaredTimes[i] == 0)) or \
                    util.manhattanDistance(newGhostStates[i].getPosition(), newPos) < 2:
                evf_result -= 100
        if len(food_distance) > 0:
            evf_result += 1 / min(food_distance)

        return evf_result


def scoreEvaluationFunction(currentGameState):
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):
        return self.value(gameState, 0)[0]

    def value(self, gameState, depth):
        if depth == self.depth * gameState.getNumAgents() \
                or gameState.isLose() or \
                gameState.isWin():
            return "", self.evaluationFunction(gameState)
        if depth % gameState.getNumAgents() != 0:
            return self.minvalue(gameState, depth)
        else:
            return self.maxvalue(gameState, depth)

    def minvalue(self, gameState, depth):
        legal_actions = gameState.getLegalActions(depth % gameState.getNumAgents())
        min_result = "", float("Inf")

        if len(legal_actions) == 0:
            return "", self.evaluationFunction(gameState)

        for action in legal_actions:
            successor = gameState.generateSuccessor(depth % gameState.getNumAgents(), action)
            result = self.value(successor, depth + 1)
            if result[1] < min_result[1]:
                min_result = (action, result[1])
        return min_result

    def maxvalue(self, gameState, depth):
        legal_actions = gameState.getLegalActions(0)
        max_result = "", float("-Inf")

        if len(legal_actions) == 0:
            return "", self.evaluationFunction(gameState)

        for action in legal_actions:
            successor = gameState.generateSuccessor(0, action)
            result = self.value(successor, depth + 1)
            if result[1] > max_result[1]:
                max_result = (action, result[1])
        return max_result


class AlphaBetaAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):

        return self.value(gameState, 0, float("-Inf"), float("Inf"))[0]

    def value(self, gameState, depth, alpha, beta):
        if depth == self.depth * gameState.getNumAgents() or \
                gameState.isWin() or\
                gameState.isLose():
            return "", self.evaluationFunction(gameState)
        if depth % gameState.getNumAgents() != 0:
            return self.minvalue(gameState, depth, alpha, beta)
        else:
            return self.maxvalue(gameState, depth, alpha, beta)

    def minvalue(self, gameState, depth, alpha, beta):
        legal_actions = gameState.getLegalActions(depth % gameState.getNumAgents())
        min_result = "", float("Inf")

        if len(legal_actions) == 0:
            return "", self.evaluationFunction(gameState)

        for action in legal_actions:
            successor = gameState.generateSuccessor(depth % gameState.getNumAgents(), action)
            result = self.value(successor, depth + 1, alpha, beta)
            if result[1] < min_result[1]:
                min_result = (action, result[1])
            if min_result[1] < alpha:
                return min_result
            beta = min(beta, min_result[1])
        return min_result

    def maxvalue(self, gameState, depth, alpha, beta):
        legal_actions = gameState.getLegalActions(0)
        max_result = "", float("-Inf")

        if len(legal_actions) == 0:
            return "", self.evaluationFunction(gameState)

        for action in legal_actions:
            successor = gameState.generateSuccessor(0, action)
            result = self.value(successor, depth + 1, alpha, beta)
            if result[1] > max_result[1]:
                max_result = (action, result[1])
            if max_result[1] > beta:
                return max_result
            alpha = max(alpha, max_result[1])
        return max_result


class ExpectimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):
        return self.value(gameState, 0)[0]

    def value(self, gameState, depth):
        if depth == self.depth * gameState.getNumAgents() or \
                gameState.isWin() or \
                gameState.isLose():
            return "", self.evaluationFunction(gameState)
        if depth % gameState.getNumAgents() != 0:
            return self.expectvalue(gameState, depth)
        else:
            return self.maxvalue(gameState, depth)

    def expectvalue(self, gameState, depth):
        legal_actions = gameState.getLegalActions(depth % gameState.getNumAgents())
        expect_value = 0

        if len(legal_actions) == 0:
            return "", self.evaluationFunction(gameState)

        weight_probability = 1. / len(legal_actions)

        for action in legal_actions:
            successor = gameState.generateSuccessor(depth % gameState.getNumAgents(), action)
            result = self.value(successor, depth + 1)
            expect_value += result[1] * weight_probability
        return "", expect_value

    def maxvalue(self, gameState, depth):
        legal_actions = gameState.getLegalActions(0)
        max_result = "", float("-Inf")

        if len(legal_actions) == 0:
            return "", self.evaluationFunction(gameState)

        for action in legal_actions:
            successor = gameState.generateSuccessor(0, action)
            result = self.value(successor, depth + 1)
            if result[1] > max_result[1]:
                max_result = (action, result[1])
        return max_result


def betterEvaluationFunction(currentGameState):
    pacman_position = currentGameState.getPacmanPosition()
    ghost_position = currentGameState.getGhostPositions()[0]
    scared_timer = currentGameState.getGhostStates()[0].scaredTimer
    ghost_distance = manhattanDistance(ghost_position, pacman_position)
    food_position = currentGameState.getFood().asList()

    food_items = []
    ghost_near = 0

    for food in food_position:
        food_items.append(-1 * manhattanDistance(pacman_position, food))
    if not food_items:
        food_items.append(0)

    if ghost_distance == 0 and scared_timer == 0:
        ghost_near = -150
    elif scared_timer > 0:
        ghost_near = -1 / ghost_distance

    num_capsules = len(currentGameState.getCapsules())

    return currentGameState.getScore() + ghost_near + num_capsules + max(food_items)


# Abbreviation
better = betterEvaluationFunction
