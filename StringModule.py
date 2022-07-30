# K
# KA
# KAM
# KAME
# KAMES
# KAMESH

name = input("enter the name : ")
length = len(name)
for row in range(length):
    for col in range(row+1):
        print(name[col], end="")
    print()