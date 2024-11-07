import random

def rps():
    number = random.randint(1, 20)
    
    while True:
        guess = int(input("Guess the random number between 1 and 20: "))
        
        if guess not in range(1, 21):
            print("Guess not in range 1-20, try again.")
        elif guess == number:
            print("Correct number!")
            break
        elif guess > number:
            print("Too high!")
        elif guess < number:
            print("Too low!")

while True:
    rps()
    cont = input("Do you want to play another round? (yes/no): ").lower()
    if cont != 'yes':
        break

