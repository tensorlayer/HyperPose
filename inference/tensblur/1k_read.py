val_path = '/Users/Joel/Desktop/1k_list.txt'
f = open(val_path)
line = f.readline()
keys=[]
while line:
    line = line.rstrip('\n')
    info = line.split()
    print(info)
    keys.append(int(info[1]))
    line = f.readline()
print(keys)