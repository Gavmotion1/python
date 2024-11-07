while True:
    try:
        starting_miles, finishing_miles = map(int, input("Enter two numbers (day one miles > 10, race day miles): ").split())
        
        if starting_miles >= 10:
            print("Starting miles:", starting_miles)
            print("Finishing miles:", finishing_miles)
            break
        else:
            print("The first number must be greater than 10. Please try again.")
    
    except ValueError:
        print("Please enter exactly two numbers. Try again.")

if starting_miles >= finishing_miles:
    print("You are already in shape for your race.")
else:
    current_miles = starting_miles  
    days = 1
    while current_miles < finishing_miles:  
        current_miles *= 1.10  
        days += 1
   
    print("It will take", days, "days to reach", finishing_miles, "miles.")
