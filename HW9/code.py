import numpy as np

centers = np.array([
	[5.3, 3.5],
	[5.1, 4.2],
	[6.0, 3.9]])
def distance(x1, y1, x2, y2):
	return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def nearest_center(point, centers):
	ans = []
	for center in centers:
		x1, y1 = point
		ans.append(distance(x1, y1, center[0], center[1]))
	return ans

while True:
	x1 = input("x1 = ")
	y1 = input("y1 = ")
	point = (float(x1), float(y1))
	print(nearest_center(point, centers))
