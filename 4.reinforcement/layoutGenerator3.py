import random
import numpy as np
import os

def createMaze(width, height):
    """ 
        Main function used to generate the maze
    """
    
    # initialize a block of "%" in the required shape
    maze = [['%' for _ in range(width)] for _ in range(height)]
    # path creation function
    subWallsForFood(maze, 1, 1)
    return maze

def subWallsForFood(maze, x, y):
    """
        Code to decide where to substitute walls with food pellets
    """

    # set down a food pellet in the current cell
    # a slight twist in x and y as this works better
    maze[y][x] = '.'
    
    directions = {}
    directions["East"] = (1, 0)
    directions["West"] = (-1, 0)
    directions["North"] = (0, 1)
    directions["South"] = (0, -1)
    
    direction_delta_list = directions.values()
    # let's randomize the order in which we fill food pellets
    random.shuffle(direction_delta_list)
    
    for delta_x, delta_y in direction_delta_list:
        
        new_x = x + delta_x
        new_y = y + delta_y
        
        # essentially replace any wall with a food pellet
        if 0 < new_x < len(maze[0]) - 1 and 0 < new_y < len(maze) - 1 and maze[new_y][new_x] == '%':
            if getNumberOfAdjacentPaths(maze, new_x, new_y, direction_delta_list) == 1:
                # recursive call to generate the maze
                subWallsForFood(maze, new_x, new_y)

def getNumberOfAdjacentPaths(maze, x, y, direction_delta_list):
    """
        This function is used to check how many free paths are there in the grid
        adjacent to the position x, y
    """

    count = 0
    for delta_x, delta_y in direction_delta_list:
        new_x = x + delta_x
        new_y = y + delta_y
        if maze[new_y][new_x] == '.':
            count += 1
    return count


def storeMaze(maze, layout_number, directory="layouts-sandbox"):
    """
        Write the maze to a file
    """

    maze_string = ""
    for row in maze:
        maze_string += ''.join(row) + '\n'
    
    file_path = os.path.join(directory, f"test_{layout_number}.lay")
    with open(file_path, "w") as file:
        file.write(maze_string)

def printMaze(maze):
    """
        Print the maze
    """
    for row in maze:
        print(''.join(row))

def getRandomNumber(low, high):
    """
        Generate a random integer number between some limits
    """

    return np.random.randint(low, high+1)

def getRandomPositon(width, height):
    """
        Get a random positon in the grid
    """

    x, y = getRandomNumber(0, width-1), getRandomNumber(0, height-1)
    return x, y

def placeFeature(maze, symbol, number):
    """
        A general function to place ghosts, power pills and pacman
    """

    i = 0
    while i < number:
        x, y = getRandomPositon(len(maze[0]), len(maze))
        if 0 <= x < len(maze) and 0 <= y < len(maze[0]):
            if maze[x][y] == ".":
                maze[x][y] = symbol
                i += 1
    return maze

def calculateGhosts(width, height):
    """
        A function to vary the number of ghosts based on the area
        of the maze to control the difficulty of the game
    """

    area = width * height
    difficulty_factor = 60
    num_ghosts = area // difficulty_factor
    return num_ghosts

for i in range(100):
    width = int(np.random.uniform(10, 20))
    height = int(np.random.uniform(7, 20))
    
    maze = createMaze(width, height)
    # placing ghosts
    maze = placeFeature(maze, "G", calculateGhosts(width, height))
    # placing power pellets
    maze = placeFeature(maze, "o", getRandomNumber(2, 4))
    # placing pacman
    maze = placeFeature(maze, "P", 1)
    # storing the maze in the layouts directory
    storeMaze(maze, i, "layouts")
