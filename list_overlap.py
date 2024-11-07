import random
list1 = random.sample(range(1, 100), 20)
list2 = random.sample(range(1, 100), 20)
duplicates = [x for x in list1 if x in list2]
print(duplicates)
