import os


f = open('kill_list.txt')

f.readline()
sh_list = []
py_list = []
for line in f:
    PID = line.strip().split()[0]
    process_type = line.strip().split()[-1]
    if process_type == 'sh':
        sh_list.append(PID)

    if process_type == 'python':
        py_list.append(PID)

for PID in sh_list:
    os.system('kill '+PID)

for PID in py_list:
    os.system('kill '+PID)
