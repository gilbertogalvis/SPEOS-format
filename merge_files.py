

# height,width = 3264, 4896


path1 = 'PART1.txt'
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
