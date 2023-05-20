import os
import time
def __read_file__(file_path):
	start = time.time()
	with open(file_path, 'r') as f:
		#print(file_path)
		data = f.readline().split()
		while not data:
			data = f.readline().split()
			if time.time() - start > 3:
				return []
		data = f.readline().split()
	f.close()
	return data

def __read_data__(data):
	PowerDC = float(data[4][:-1])
	PowerDC_unit = data[4][-1]
	if PowerDC_unit == "u":
		PowerDC = PowerDC
	elif PowerDC_unit == "n":
		PowerDC = PowerDC * 1e-3
	elif PowerDC_unit == "m":
		PowerDC = PowerDC * 1e3

	GBW = float(data[5][:-1])
	GBW_unit = data[5][-1]
	if GBW_unit == "M":
		GBW = GBW
	elif GBW_unit == "K":
		GBW = GBW * 1e-3
	elif GBW_unit == "G":
		GBW = GBW * 1e3
	GBW = -GBW

	RmsNoise = float(data[6][:-1])
	RmsNoise_unit = data[6][-1]
	if RmsNoise_unit == "n":
		RmsNoise = RmsNoise
	elif RmsNoise_unit == "p":
		RmsNoise = RmsNoise * 1e-3
	elif RmsNoise_unit == "u":
		RmsNoise = RmsNoise * 1e3


	SettlingTime = float(data[7][:-1])
	SettlingTime_unit = data[7][-1]
	if SettlingTime_unit == "u":
		SettlingTime = SettlingTime
	elif SettlingTime_unit == "n":
		SettlingTime = SettlingTime * 1e-3
	elif SettlingTime_unit == "m":
		SettlingTime = SettlingTime * 1e3

	return [PowerDC, GBW, RmsNoise, SettlingTime]
folder_path = "../data"
max_list = [-99999999, -99999999, -99999999, -99999999]
min_list = [9999999, 9999999, 9999999, 9999999]
t = 0
for filename in os.listdir(folder_path):
	if filename.endswith(".txt"):
		t += 1
		file_path = os.path.join(folder_path, filename)
		data = __read_file__(file_path)
		if not data:
			continue
		output = __read_data__(data)
		for i in range(len(output)):
			max_list[i] = max(max_list[i], output[i])
			min_list[i] = min(min_list[i], output[i])
	if t % 1000 == 0:
		print(max_list)
		print(min_list)
print(max_list)
print(min_list)