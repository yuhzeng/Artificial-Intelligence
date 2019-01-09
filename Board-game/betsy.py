#!/usr/bin/env python3
# betsy.py 
#
"""
Created on Mon Oct 15 12:58:51 2018

@author: Yuhan Zeng
@Teammates: Hu Hai, Yuhan Zeng
"""

import sys
import math
import copy

# Convert a string representation of a state to a (n+3)*n board representation
def stringToBoard(stateString):
    board = []
    for row in range(n + 3): # n + 3 rows
        board.append([stateString[col] for col in range(row*n, (row+1)*n)])
    return board

# Convert a (n+3)*n board representation of a state to string representation 
def boardToString(stateBoard):
    string = ''
    for row in range(n + 3): # n + 3 rows
        for col in range(n): # n columns
            string += stateBoard[row][col]
    return string

# drop function does not check if the column is full
def drop(board, col, player): # player = either 'x' or 'o'
    # Make a copy of the board, and do not change the original board
    boardCopy = copy.deepcopy(board)
    for i in range(n+3-1, -1, -1): # Traverse the column from bottom to top
        if boardCopy[i][col] == '.':
            boardCopy[i][col] = player
            break
    return boardCopy

# rotate function does not check if the column is empty
def rotate(board, col):
    # Make a copy of the board, and do not change the original board
    boardCopy = copy.deepcopy(board)
    bottom = boardCopy[n+2][col]
    for i in range(n+3-1, -1, -1):
        if i == 0 or boardCopy[i - 1][col] == '.':
            break
        boardCopy[i][col] = boardCopy[i - 1][col]
    boardCopy[i][col] = bottom
    return boardCopy

# successors function returns all the valid successors by a drop or a rotate
# return value is a list of (move, successor_board) tuples, e.g., move 3 strands 
# for dropping at column 3, -1 stands for rotating column 1 (column index starts from 1)
def successors(board, player):
    succ = []
    for col in range(n):
        if board[0][col] == '.': # When this column is not full, drop a pebble
            succ.append((col + 1, drop(board, col, player)))
        if board[n+2][col] != '.': # When this column is not empty, rotate it
            succ.append((-(col + 1), rotate(board, col)))
    return succ
        
# Heuristic function 1: e(s) = (number of x's - number of o's) in the top n rows
# When a player is about to win, it should occupy more tiles in the upper n rows
def heuristic1(board):
    numX = 0 
    numO = 0
    for row in range(n):
        for col in range(n):
            if board[row][col] == 'x':
                numX += 1
            elif board[row][col] == 'o':
                numO += 1
    return numX - numO

# Check if there is a win/lose: n^2 means player x wins, -n^2 means player o wins, and the heuristic value for no win.
# player determines  which player is checked first in case that the board has a row, column or diagonal of n pebbles for both players
# The function returns +/-n^2 for a win/lose, a much larger value than the heuristic function value of any node
def isTerminal(board, player):
    # Check each row
    oneRowOfX = 0
    oneRowOfO = 0
    for row in range(n):
        numX = 0
        numO = 0
        for col in range(n):
            if board[row][col] == '.':
                break
            elif board[row][col] == 'x':
                numX += 1
            else:
                numO += 1
        if numX == n:
            oneRowOfX += 1
        if numO == n:
                oneRowOfO += 1
    if player == 'x':
        if oneRowOfX != 0:
            return winValue
        elif oneRowOfO != 0:
            return -winValue
    else:
        if oneRowOfO != 0:
            return -winValue
        elif oneRowOfX != 0:
            return winValue
    # Check each column
    oneColOfX = 0
    oneColOfO = 0
    for col in range(n):
        numX = 0
        numO = 0
        for row in range(n):
            if board[row][col] == '.':
                break
            elif board[row][col] == 'x':
                numX += 1
            else:
                numO += 1
        if numX == n:
            oneColOfX += 1
        if numO == n:
            oneColOfO += 1
    if player == 'x':
        if oneColOfX != 0:
            return winValue
        elif oneColOfO != 0:
            return -winValue
    else:
        if oneColOfO != 0:
            return -winValue
        elif oneColOfX != 0:
            return winValue

    # Check prime diagonal
    numX = 0
    numO = 0
    for row in range(n):
        col = row
        if board[row][col] == '.':
            break
        elif board[row][col] == 'x':
            numX += 1
        else:
            numO += 1
    if player == 'x':
        if numX == n:
            return winValue
        if numO == n:
            return -winValue
    else:
        if numO == n:
            return -winValue
        if numX == n:
            return winValue
    # Check secondary diagonal
    numX = 0
    numO = 0
    for row in range(n):
        col = (n - 1) - row
        if board[row][col] == '.':
            break
        elif board[row][col] == 'x':
            numX += 1
        else:
            numO += 1
    if player == 'x':
        if numX == n:
            return winValue
        if numO == n:
            return -winValue
    else:
        if numO == n:
            return -winValue
        if numX == n:
            return winValue
    # When no one wins
    return heuristic1(board)

# maxValue() returns the (alpha, bestMove, bestSucc) tuple at a Max node
def maxValue(board, alpha, beta, currDepth, maxDepth):
    terminal = isTerminal(board, 'x') 
    if abs(terminal) == winValue or currDepth == maxDepth: 
        return (terminal, 0, board)
    bestVal = -math.inf
    bestSucc = None
    bestMove = 0
    for (move, successor) in successors(board, 'x'):
        (child_beta, child_move, child_succ) = minValue(successor, alpha, beta, currDepth + 1, maxDepth)
        if currDepth == 0 and child_beta > bestVal:
            bestVal = child_beta
            bestSucc = successor
            bestMove = move
        alpha = max(alpha, child_beta)
        if alpha >= beta:
            break #return (alpha, move, successor)
    return (alpha, bestMove, bestSucc)

# minValue() returns the (beta, bestMove, bestSucc) tuple at a Min node
def minValue(board, alpha, beta, currDepth, maxDepth):
    terminal = isTerminal(board, 'o') 
    if abs(terminal) == winValue or currDepth == maxDepth: 
        return (terminal, 0, board)
    bestVal = math.inf
    bestSucc = None
    bestMove = 0 
    for (move, successor) in successors(board, 'o'):
        (child_alpha, child_move, child_succ) = maxValue(successor, alpha, beta, currDepth + 1, maxDepth)
        if currDepth == 0 and child_alpha < bestVal:
            bestVal = child_alpha
            bestSucc = successor
            bestMove = move
        beta = min(beta, child_alpha)
        if alpha >= beta:
            break #return (beta, move, successor)
    return (beta, bestMove, bestSucc)

# returns the (move, successor) tuple which is the best up to the deepest horizon searched so far
def alphaBetaDecision(board, player, alpha, beta, maxDepth):
    if player == 'x':
        (alpha, move, successor) = maxValue(board, alpha, beta, 0, maxDepth)
    else:
        (beta, move, successor) = minValue(board, alpha, beta, 0, maxDepth)   
    return (move, successor)

n = int(sys.argv[1]) 
startPlayer = str(sys.argv[2])
startString = str(sys.argv[3])
timeLimit = int(sys.argv[4])

startBoard = stringToBoard(startString)
horizon = 1
winValue = 999

# Increment the search depth (horizon) by 1 each time until a win/lose state is reached.
while(1):
    (move, successor) = alphaBetaDecision(startBoard, startPlayer, -math.inf, math.inf, horizon)
    print('After searching depth of', horizon, 'I would recommend the following move:')
    print(move, boardToString(successor))
    horizon += 1
    
        
        
    
    
    
    
    
    
    
    