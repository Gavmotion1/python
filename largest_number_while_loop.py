playing = True
numbers = []

while playing:
    integer = int(input("input numbers and 0 to end program and print highest number "))
    numbers.append(integer)
    print(numbers)
    if integer == 0:
        print("the highest number was " + str(max(numbers)))
        playing = False
