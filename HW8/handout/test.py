def fn(x, reward, new_x):
	return x + 0.01 * (reward + new_x - x)
x_1 = 0.6
x_2 = -0.3
x_3 = -0.5
x_4 = 0.8

for i in range(10000):
	#x_2 = fn(x_2, 1, 0)
	#x_3 = fn(x_3, -1, 0)
	x_1 = fn(x_1, 0, x_4)
	x_4 = fn(x_4, 0, 0)
	print(x_1)
	print(x_2)
	print(x_3)
	print(x_4)
	print("---------------")
