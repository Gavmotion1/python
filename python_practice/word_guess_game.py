import random
words = [
    ("Python", "A popular programming language."),
    ("Algorithm", "A step-by-step procedure for solving a problem."),
    ("Function", "A block of code that only runs when it is called."),
    ("Variable", "A container for storing data values."),
    ("Integer", "A whole number without a decimal."),
    ("Syntax", "The set of rules that defines the combinations of symbols."),
    ("String", "A sequence of characters."),
    ("Boolean", "A type that can be either True or False."),
    ("Compile", "The process of converting code into machine language."),
    ("Debug", "The process of finding and fixing errors in code."),
    ("Array", "A collection of elements of the same type."),
    ("Loop", "A sequence of instructions that repeats until a condition is met."),
    ("Binary", "A base-2 number system."),
    ("Exception", "An error that occurs during the execution of a program."),
    ("Recursion", "A function that calls itself."),
    ("Module", "A file containing Python code."),
    ("Tuple", "An immutable sequence of Python objects."),
    ("Dictionary", "A collection of key-value pairs."),
    ("List", "A collection of elements that can be of different types."),
    ("Queue", "A collection of elements that follows the FIFO principle."),
    ("Stack", "A collection of elements that follows the LIFO principle."),
    ("Object", "An instance of a class."),
    ("Class", "A blueprint for creating objects."),
    ("Inheritance", "A way to form new classes using classes that have already been defined."),
    ("Encapsulation", "The bundling of data and methods that operate on the data."),
    ("Polymorphism", "The ability to use a common interface for multiple forms."),
    ("Abstraction", "The concept of hiding the complex implementation details."),
    ("Lambda", "An anonymous function defined with the lambda keyword."),
    ("Iterator", "An object that allows traversal through a container."),
    ("Generator", "A function that returns an iterator.")
]

# Initialize best score
best_score = float('inf')

def word_guess():
    global best_score
    while True:
        random_word, hint = random.choice(words)
        random_word = random_word.upper()
        length = len(random_word)
        word = "__ " * length
        word = word.rstrip()  # Remove trailing space

        guessed_letters = set()
        incorrect_guesses = 0
        max_incorrect = 6  # Max number of incorrect guesses allowed
        guess_count = 0

        while "__ " in word:
            print("\nWord to guess:", word.center(77))
            print("Guessed letters:", ', '.join(sorted(guessed_letters)))
            print("Number of guesses made:", guess_count)
            print("Incorrect guesses remaining:", max_incorrect - incorrect_guesses)
            print("Hint:", hint)

            guess = input('Guess a letter or word: ').upper()
            guess_count += 1

            if guess.isalpha():
                if len(guess) > 1:  # Full word guess
                    if guess == random_word:
                        word = random_word
                        print("Congratulations! You've guessed the word:", word)
                        break
                    else:
                        print("Incorrect guess. Try again.")
                        incorrect_guesses += 1
                else:  # Single letter guess
                    if guess in guessed_letters:
                        print("You've already guessed that letter.")
                    else:
                        guessed_letters.add(guess)
                        if guess in random_word:
                            word_list = list(word)
                            for i, char in enumerate(random_word):
                                if char == guess:
                                    index = i * 3
                                    word_list[index] = char
                            word = "".join(word_list)
                        else:
                            print("Incorrect letter.")
                            incorrect_guesses += 1
            else:
                print("Invalid input. Please enter a letter or word.")

            if incorrect_guesses >= max_incorrect:
                print("Game over! The word was:", random_word)
                break

        # Update best score
        if guess_count < best_score:
            best_score = guess_count
            print("New best score:", best_score)

        # Option to play again
        play_again = input("\nDo you want to play again? (yes/no): ").strip().lower()
        if play_again != 'yes':
            print("Thanks for playing!")
            break

word_guess()