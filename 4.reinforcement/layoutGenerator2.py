import random
import numpy as np

def createMaze(width, height):
    maze = [['%' for _ in range(width)] for _ in range(height)]
    divide(maze, 0, 0, width - 1, height - 1)
    return maze

def divide(maze, x1, y1, x2, y2):
    if x2 - x1 < 2 or y2 - y1 < 2:
        return

    if x2 - x1 > y2 - y1:
        divide_vertically(maze, x1, x2, y1, y2)
    else:
        divide_horizontally(maze, x1, x2, y1, y2)

def divide_vertically(maze, x1, x2, y1, y2):
    wall_x = random.randint(x1 + 1, x2 - 1)
    passage_y = random.randint(y1, y2)
    for y in range(y1, y2 + 1):
        if y != passage_y:
            maze[y][wall_x] = '.'
    divide(maze, x1, y1, wall_x - 1, y2)
    divide(maze, wall_x + 1, y1, x2, y2)

def divide_horizontally(maze, x1, x2, y1, y2):
    wall_y = random.randint(y1 + 1, y2 - 1)
    passage_x = random.randint(x1, x2)
    for x in range(x1, x2 + 1):
        if x != passage_x:
            maze[wall_y][x] = '.'
    divide(maze, x1, y1, x2, wall_y - 1)
    divide(maze, x1, wall_y + 1, x2, y2)

def printMaze(maze):
    for row in maze:
        print(''.join(row))

def enclose_maze(maze, width, height):
    new_width = width + 2
    height = height + 2
    
    for row in maze:
        row.insert(0, "%")
        row.append("%")
    
    maze.insert(0, ['%' for _ in range(new_width)])
    maze.append(['%' for _ in range(new_width)])

    return maze


import os

def storeMaze(maze, layout_number, directory="layouts-bkp"):
    maze_string = ""
    for row in maze:
        maze_string += ''.join(row) + '\n'
    
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Write maze to a file
    file_path = os.path.join(directory, f"test_{layout_number}.lay")
    with open(file_path, "w") as file:
        file.write(maze_string)



def getRandomNumber(low, high):
    return np.random.randint(low, high+1)

def getRandomPositon(width, height):
    x, y = getRandomNumber(0, width-1), getRandomNumber(0, height-1)
    return x, y

irreplacableList = ["P", "G", "%"]

def placeFeature(maze, symbol, number):
    i = 0
    while i < number:
        x, y = getRandomPositon(len(maze[0]), len(maze))
        if 0 <= x < len(maze) and 0 <= y < len(maze[0]):
            if maze[x][y] == ".":
                maze[x][y] = symbol
                i += 1
    return maze

for i in range(100):
    width = np.random.uniform(10, 20)
    height = np.random.uniform(12, 20)
    
    maze = createMaze(int(width), int(height))
    maze = placeFeature(maze, "G", getRandomNumber(2, 4))
    maze = placeFeature(maze, "o", getRandomNumber(2, 4))
    maze = placeFeature(maze, "P", 1)
    
    maze = enclose_maze(maze, len(maze[0]), len(maze))
    
    storeMaze(maze, i, "layouts")
