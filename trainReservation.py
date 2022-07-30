# get the confirmation of Travelling (Variable)
# If Travel is Yes then Ask for number of people
# get the details of Male and Female

travelling = input('Yes or No')
while travelling in ('y', 'yes', 'Y', 'Yes'):
    num = int(input("Enter the number of persons : "))
    for num in range(1, num + 1):
        name = input("Name :")
        age = input("Age :")
        sex = input("Male or Female :")
        print(name)
        print(age)
        print(sex)
    travelling = input("oops! Forgot to add anyone ? ")