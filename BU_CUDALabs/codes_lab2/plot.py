
with open('temperature_cpu.txt', 'r') as fp:
    data = [list(line.strip().split('\t')) for line in fp]

data.remove([''])

print([elem[0] for elem in data])
# print([elem[1] for elem in data])
# print([elem[2] for elem in data])
