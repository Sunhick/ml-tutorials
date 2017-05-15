#!/usr/bin/python3

"""
Statistics and probability
"""

import sys
import random
import numpy as np

class MontyHall(object):
    def  __init__(self):
        self.changeCount = 0
        self.noChangeCount = 0
        self.totalTrails = 0
        self.doors = ["car", "goat", "goat"]

    def initGame(self):
        np.random.shuffle(self.doors)

    def play(self, pickDoor):
        self.totalTrails += 1
        options = [i for i in range(0, 3) if self.doors[i] != "car" and i != pickDoor]
        openDoor = np.random.choice(options, 1)[0]

        if self.doors[pickDoor] == "car":
            self.noChangeCount += 1

        nPickDoor = list(set(range(0,3)) - set([pickDoor, openDoor]))[0]
        if self.doors[nPickDoor] == "car":
            self.changeCount += 1

    def printStats(self):
        print("Wins by not changing =", self.noChangeCount/self.totalTrails)
        print("Wins by changing =", self.changeCount/self.totalTrails)


def main(args):
    m = MontyHall()
    N = int(args[0])

    for i in range(0, N):
        m.initGame()
        pick = np.random.choice(range(0,3), 1)[0]
        m.play(pick)

    m.printStats()

if __name__ == "__main__":
    main(sys.argv[1:])