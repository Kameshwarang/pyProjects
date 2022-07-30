# Pyramid

def pattern(n):
    k = 2 * n - 2
    print("Your K's value is : ", k)
    for i in range(0, n-1):
        print("Your I's value is : ", i)
        for j in range(0, k):
            print("Your J's value is : ", j)
            print(end=" ")
        k = k - 1
        for j in range(0, i + 1):
            print("* ", end="")
        print("\r")

pattern(5)
