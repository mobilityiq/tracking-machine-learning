from enum import Enum


# Define the transportation mode Enum
class TransportationMode(Enum):
    DRIVING = 'driving'
    CYCLING = 'cycling'
    TRAIN = 'train'
    BUS = 'bus'
    SUBWAY = 'metro'
    RUNNING = 'running'
    WALKING = 'walking'
    STATIONARY = 'stationary'