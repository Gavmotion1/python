def unique_in_order(sequence):
    if not sequence:
        return []

    result = []
    previous = None

    for item in sequence:
        if item != previous:
            result.append(item)
        previous = item

    return result



def count_divisors(n):
    count = 0
    for i in range(1, n + 1):  # Include n in the range
        if n % i == 0:
            count += 1
    return count



def queue_time(customers, n):
    qn = [0] * n  # Initialize a list to track the total time of each cash register
    for c in customers:  # Iterate over each customer
        qn = sorted(qn)  # Sort the list to find the register with the least load
        qn[0] += c  # Assign the customer to the register with the least load
    return max(qn)  # Return the maximum time from the registers, representing the total time



def digitize(n):
    # Convert the number to a string, then to a list of characters
    r = list(str(n))
    # Reverse the list of characters
    r.reverse()
    # Convert each character back to an integer
    r = [int(digit) for digit in r]
    return r



def points(games):
    total_points = 0
    for game in games:
        x, y = map(int, game.split(':'))
        if x > y:
            total_points += 3
        elif x == y:
            total_points += 1
    return total_points



def find_short_length(s):
    # Split the string into a list of words
    words = s.split()
    # Find the length of the shortest word
    return len(min(words, key=len))



def fake_bin(x):
    # Use a list comprehension to transform each character
    return ''.join('1' if int(char) >= 5 else '0' for char in x)



def validate_pin(pin):
    if pin.isnumeric() and len(pin) in [4, 6]:
        return True
    else:




        return False



def min_max(lst):
    # Check if the list is empty; if so, return an empty list
    if not lst:
        return []
    
    # Calculate the minimum and maximum values
    min_val = min(lst)
    max_val = max(lst)
    
    # Return the result as a list with the minimum and maximum values
    return [min_val, max_val]



def is_palindrome(s):
    s = s.lower()  # Convert the string to lowercase
    return s == s[::-1]



def filter_list(l):
    filtered = []
    for item in l:
        if type(item) != str:  # Check if the item is not a string
            filtered.append(item)  # Add non-string items to the list
    return filtered


def = data:
    return ["Senior" if age >= 55 and handicap >= 8 else "Open" for (age, handicap) in data]



def areYouPlayingBanjo(name):
    if name[0].lower() == 'r':
        return name + ' plays banjo'
    else:
        return name + ' does not play banjo'



def xo(s):
    s = s.lower()
    return s.count('x') == s.count('o')



price = 49
txt = f"It is very {'Expensive' if price>50 else 'Cheap'}"




def calculate_average():
    total_sum = 0
    count = 0

    while True:
        try:
            number = int(input("Enter an integer (0 to stop): "))
            
            if number == 0:
                if count == 0:
                    print("No numbers were entered.")
                else:
                    average = total_sum / count
                    print(f"The average is: {average:.1f}")
                break
            
            total_sum += number
            count += 1
        
        except ValueError:
            print("Invalid input. Please enter an integer.")

calculate_average()



def divide_numbers(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        print("Cannot divide by zero!")
        return None
    except TypeError:
        print("Both inputs must be numbers.")
        return None
    else:
        print("Division successful!")
        return result
    finally:
        print("Execution completed.")

# Test the function
print(divide_numbers(10, 2))  # This will execute the try, else, and finally blocks
print(divide_numbers(10, 0))  # This will execute the try, except, and finally blocks
print(divide_numbers(10, "a"))  # This will execute the try, except, and finally blocks



def check_age(age):
    if age < 18:
        raise ValueError("Age must be at least 18.")
    print("Age is valid.")

try:
    check_age(15)
except ValueError as e:
    print(f"Error: {e}")
