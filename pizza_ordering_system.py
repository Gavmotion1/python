print(" Welcome to Nicco's Pizza Order Planner ")
people_count = int(input("How many people are in your party? "))

large_pizzas = people_count // 7
remaining_people = people_count % 7

medium_pizzas = remaining_people // 3
remaining_people = remaining_people % 3

small_pizzas = remaining_people // 1
remaining_people = remaining_people % 1

print(f"you will need {large_pizzas} large pizzas, {medium_pizzas} medium pizzas, and {small_pizzas} small pizzas")


print("Enjoy your pizzas")
