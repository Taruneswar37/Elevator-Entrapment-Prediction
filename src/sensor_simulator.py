import random
import numpy as np

class ElevatorSensor:

    def __init__(self):
        self.vibration = 4.0
        self.temp = 35
        self.doors = 120
        self.usage = 50
        self.speed = 1.0

    def next(self):
        # normal behaviour
        self.vibration += random.uniform(-0.3, 0.3)
        self.temp += random.uniform(-0.5, 0.5)
        self.doors += random.randint(0, 3)
        self.usage += random.uniform(-2, 2)

        # RANDOM FAILURE PATTERNS
        if random.random() < 0.08:
            # door issue
            self.vibration += random.uniform(2, 4)
            self.temp += 3

        if random.random() < 0.05:
            # overload
            self.speed -= 0.3
            self.temp += 4

        return {
            "vibration": round(abs(self.vibration),2),
            "usage_rate": round(abs(self.usage),2),
            "door_cycles": int(self.doors),
            "speed": round(abs(self.speed),2),
            "temperature": round(abs(self.temp),2)
        }
