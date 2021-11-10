# import pandas as pd
import numpy as np

# path1 = '/home/jorge/work/speos/GUI_v5.2/a_speos_data/X.npy'

# path2 = '/home/jorge/work/speos/GUI_v5.2/a_speos_data/Y.npy'

# path3 = '/home/jorge/work/speos/GUI_v5.2/a_speos_data/Z.npy'


def RebuildSpeosTxt(path1, path2, path3, pathout):

	with open(path1, 'rb') as f:
		X = np.load(f)

	with open(path2, 'rb') as f:
		Y = np.load(f)

	with open(path3, 'rb') as f:
		Z = np.load(f)

	height,width = Z.shape[0], Z.shape[1]
	data = np.zeros([height,width,3])
	new_line = ''

	print ("Doing part1")
	text_file = open("part1.txt", "w")
	text_file.write("")
	text_file.close()
	new_line1 = ''

	xi = 0
	for i in range(0, height):

		new_line1 = ''
		for yi in range(0,width-1):

			new_line1 = new_line1 + str(X[i,yi])+' ' + str(Y[i,yi]) +' '+ str(0) +' '+ str(Z[i,yi]) +' '
			Cx = X[i,yi]/(X[i,yi]+Y[i,yi]+Z[i,yi])
			Cy = Y[i,yi]/(X[i,yi]+Y[i,yi]+Z[i,yi])

			data[xi, yi] = np.array([xi, yi, Y[i,yi]])

		xi = xi +1
		new_line1 = new_line1 +"\n"	
		file_object = open('part1.txt', 'a')	
		file_object.write(new_line1)

	with open('part1.txt') as f:
	    part1 = f.readlines()

	print ("len(part1)", len(part1))

	print ("Data loaded #")
	print ("width", width)
	print ("height", height)

	delta = 0.1
	minX = (width/2)*(-0.1)
	maxX = (width/2)*(0.1)

	minY = (height/2)*(-0.1)
	maxY = (height/2)*(0.1)
	print ("minX",minX, "minY", minY)
	print ("Doing part2")

	part2 = "X Y value \n"
		
	print ("data.shape#",data.shape)
	height , width =  data.shape[0:2]

	print ("height , width", height , width)

	print (data[height-1, width-1])

	for i in range(0,height):	
		for j in range(0,width):
			data[i, j][0] = (minX + 0.1*j)
			data[i, j][1] = (minY + 0.1*i)

			xc = data[i, j][0]
			yc = data[i, j][1]
			xc = round(xc,1)
			yc = round(yc,1)		
			part2 = str(xc)+' '+ str(yc)+' '+ str(data[height-1-i, j][2])+"\n"		        
			with open("part2.txt", "a") as file_object:
				file_object.write(part2)

	print ("Saved file#")

	############################################ MERGE FILES ########################################################

	path1 = 'part1.txt'
	path2 = 'part2.txt'

	with open(path1) as f:
	    part1 = f.readlines()

	print ("Part1 len", len(part1))
	with open(path2) as f:
	    part2 = f.readlines()
	print ("Part2 len", len(part2))



	head_text = ("2\n 2 \n 1\n 1\n -244.8000000000	244.8000000000	-163.2000000000	163.2000000000\n 2800	2000\n -1\n SeparatedByLayer 1	1	1\n No name\n")

	# part1 = lines1
	# part2 = lines2

	with open("speos_format.txt", "a") as file_object:
		file_object.write(head_text)
	print ("created head")
	part1 = open(path1, 'r').read()
	part2 = open(path2, 'r').read()

	with open("speos_format.txt", "a") as file_object:
		file_object.write(part1)

	print ("Part1 added")

	with open("speos_format.txt", "a") as file_object:
		file_object.write("X Y Value\n")

	print ("Part2 Separation")

	with open("speos_format.txt", "a") as file_object:
		file_object.write(part2)
	print ("Part2 added")
