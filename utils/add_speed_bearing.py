import numpy as np
import math

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Radius of Earth in meters
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
        math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance
    
def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the bearing between two points on the earth specified in decimal degrees.
    Returns bearing in degrees (0 to 360).
    """
    dLon = math.radians(lon2 - lon1)
    y = math.sin(dLon) * math.cos(math.radians(lat2))
    x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dLon)
    initial_bearing = math.degrees(math.atan2(y, x))
    # Normalize bearing to lie between 0-360 degrees
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing


# Load the data
# data = np.loadtxt("../training/files/SHL-2023/validate/Hand/Location.txt")

with open("../training/files/SHL-2023/Hand/Location.txt", "r") as file:
    lines = file.readlines()

# timestamps = data[:, 0]
# latitudes = data[:, 4]
# longitudes = data[:, 5]

speeds = [0]  # Initial speed
bearings = [0]  # Initial bearing

TIME_THRESHOLD = 30 * 1000  # 30 seconds in milliseconds


# Splitting the data and computing speeds and bearings
for i in range(1, len(lines)):
    prev_data = lines[i-1].strip().split()
    curr_data = lines[i].strip().split()
    
    prev_timestamp, prev_latitude, prev_longitude = float(prev_data[0]), float(prev_data[4]), float(prev_data[5])
    curr_timestamp, curr_latitude, curr_longitude = float(curr_data[0]), float(curr_data[4]), float(curr_data[5])
    
    dt = curr_timestamp - prev_timestamp  # Time difference in milliseconds
    
    if dt == 0 or dt > TIME_THRESHOLD:
        speed = -1
        bearing = -1
    else:
        distance = haversine(prev_latitude, prev_longitude, curr_latitude, curr_longitude)
        speed = distance / (dt / 1000)  # Convert dt to seconds
        bearing = calculate_bearing(prev_latitude, prev_longitude, curr_latitude, curr_longitude)
    
    speeds.append(speed)
    bearings.append(bearing)


# Save the new data to the same file or a different one
# np.savetxt("Location_with_speed_bearing.txt", new_data)

# Writing to a new file with the calculated speeds and bearings
with open("Location_with_speed_bearing.txt", "w") as out_file:
    for i, line in enumerate(lines):
        out_file.write(f"{line.strip()} {speeds[i]:.8f} {bearings[i]:.8f}\n")

print("Speed and bearing added and saved to 'Location_with_speed_bearing.txt'")