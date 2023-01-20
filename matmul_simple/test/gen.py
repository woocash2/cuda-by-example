import random
n = int(input())
print(n)
for i in range(n):
    for j in range(n):
        print(random.randint(-100, 100), end=' ')
    print()
for i in range(n):
    for j in range(n):
        print(random.randint(-100, 100), end=' ')
    print()