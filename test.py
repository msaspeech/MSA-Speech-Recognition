

dataset = []
for i in range(0,88):
    dataset.append(i)

first_interval = int(len(dataset) / 4)
second_interval = int(len(dataset) / 2)
third_interval = int(len(dataset) * 3 / 4)

print(0, int(first_interval),int(first_interval)+1, int(second_interval), int(second_interval)+1, int(third_interval))
dataset1 = dataset[0:first_interval]
dataset2 = dataset[first_interval : second_interval]
dataset3 = dataset[second_interval : third_interval]
dataset4 = dataset[third_interval : len(dataset)]

del dataset

bigger = [dataset1, dataset2, dataset3, dataset4]
print(bigger[0])
del bigger[0]
print(bigger[0])
print(bigger)