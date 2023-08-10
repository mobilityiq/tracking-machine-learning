from enum import Enum


# Define the transportation mode Enum
class TransportationMode(Enum):
    UNKNOWN = 'unknown'
    STATIONARY = 'stationary'
    WALKING = 'walking'
    RUNNING = 'running'
    CYCLING = 'cycling'
    DRIVING = 'driving'
    BUS = 'bus'
    TRAIN = 'train'
    SUBWAY = 'metro'
