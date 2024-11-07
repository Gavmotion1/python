### imports

import random

### State Variables
year = 0
population = 95
bushels_in_store = 2800
acres_owned = 3000
bushels_eaten_by_rats = acres_owned - bushels_in_store
harvest_per_acre = 3
acres_per_person = acres_owned / harvest_per_acre
new_immigrants = 5
user_input = 1
total_number_starved = 0
percent_of_population_that_starved = 0


#### Game loop
while year < 11:
    print(f"\nHAMURABI: I BEG TO REPORT TO YOU, IN YEAR {year}, {total_number_starved} PEOPLE STARVED, {new_immigrants} CAME TO THE CITY.")
    population += new_immigrants
    
    if user_input <= 0:
        population = population // 2
        print("A HORRIBLE PLAGUE STRUCK! HALF THE PEOPLE DIED.")
    
    print(f"POPULATION IS NOW {population}")
    print(f"THE CITY NOW OWNS {acres_per_person} ACRES.")
    print(f"YOU HARVESTED {harvest_per_acre} BUSHELS PER ACRE.")
    print(f"THE RATS ATE {bushels_eaten_by_rats} BUSHELS.")
    print(f"YOU NOW HAVE {bushels_in_store} BUSHELS IN STORE.\n")
    
    if year == 10:
        break
    
    random_variable = random.randint(1, 10)
    harvest_per_acre = random_variable + 17
    print(f"LAND IS TRADING AT {harvest_per_acre} BUSHELS PER ACRE.")
    
    while True:
        user_input = int(input("HOW MANY ACRES DO YOU WISH TO BUY? "))
        if user_input < 0:
            exit("Game Over!")
        if harvest_per_acre * user_input <= bushels_in_store:
            break
        else:
            print("HAMURABI: THINK AGAIN. YOU DON'T HAVE ENOUGH BUSHELS.")
    
    if user_input == 0:
        continue
    
    acres_per_person += user_input
    bushels_in_store -= harvest_per_acre * user_input
    random_variable = 0
    
    while True:
        user_input = int(input("HOW MANY ACRES DO YOU WISH TO SELL? "))
        if user_input < 0:
            exit("Game Over!")
        if user_input <= acres_per_person:
            break
        else:
            print("HAMURABI: THINK AGAIN. YOU CAN'T SELL MORE ACRES THAN YOU OWN.")
    
    acres_per_person -= user_input
    bushels_in_store += harvest_per_acre * user_input
    random_variable = 0
    
    while True:
        user_input = int(input("HOW MANY BUSHELS DO YOU WISH TO FEED YOUR PEOPLE? "))
        if user_input < 0:
            exit("Game Over!")
        if user_input <= bushels_in_store:
            break
        else:
            print("HAMURABI: THINK AGAIN. YOU DON'T HAVE ENOUGH BUSHELS.")
    
    bushels_in_store -= user_input
    random_variable = 1
    
    while True:
        acres_to_plant = int(input("HOW MANY ACRES DO YOU WISH TO PLANT WITH SEED? "))
        if acres_to_plant == 0:
            break
        if acres_to_plant < 0:
            exit("Game Over!")
        if acres_to_plant <= acres_per_person:
            break
        else:
            print("HAMURABI: THINK AGAIN. YOU CAN'T PLANT MORE ACRES THAN YOU OWN.")
    
    if acres_to_plant == 0:
        continue
    
    if acres_to_plant // 2 > bushels_in_store:
        print("HAMURABI: THINK AGAIN. YOU DON'T HAVE ENOUGH BUSHELS FOR SEED.")
        continue
    
    if acres_to_plant >= 10 * population:
        print(f"BUT YOU HAVE ONLY {population} PEOPLE TO TEND THE FIELDS! NOW THEN,")
        continue
    
    bushels_in_store -= acres_to_plant // 2
    
    random_variable = random.randint(1, 5)
    
    if random_variable % 2 == 0:
        bushels_eaten_by_rats = bushels_in_store // random_variable
    
    bushels_in_store = bushels_in_store - bushels_eaten_by_rats + (acres_to_plant * harvest_per_acre)
    
    if random_variable % 2 == 0:
        bushels_eaten_by_rats = bushels_in_store // random_variable
    
    bushels_in_store = bushels_in_store - bushels_eaten_by_rats + (acres_to_plant * harvest_per_acre)
    
    new_immigrants = int(random_variable * (20 * acres_per_person + bushels_in_store) / population / 100 + 1)
    
    random_variable = user_input // 20
    
    if random.randint(1, 100) >= 15:
        user_input = int(10 * (2 * random.random() - 0.3))
    
    if population <= random_variable:
        acres_to_plant = population - random_variable
        if acres_to_plant > 0.45 * population:
            percent_of_population_that_starved = ((year - 1) * percent_of_population_that_starved + acres_to_plant * 100 / population) / year
        population = random_variable
        total_number_starved += acres_to_plant
    else:
        print(f"YOU STARVED {population - random_variable} PEOPLE IN ONE YEAR!!!")
        print("DUE TO THIS EXTREME MISMANAGEMENT YOU HAVE NOT ONLY")
        print("BEEN IMPEACHED AND THROWN OUT OF OFFICE BUT YOU HAVE")
        print("ALSO BEEN DECLARED NATIONAL FINK!!!!")
        exit()
    
    year += 1

print(f"IN YOUR 10-YEAR TERM OF OFFICE, {percent_of_population_that_starved:.2f} PERCENT OF THE POPULATION STARVED PER YEAR ON AVERAGE, A TOTAL OF {total_number_starved} PEOPLE DIED!!")
sustainability = acres_per_person / population
print(f"YOU STARTED WITH 10 ACRES PER PERSON AND ENDED WITH {L:.2f} ACRES PER PERSON.\n")

if percent_of_population_that_starved > 33 or sustainability  < 7:
    print("YOUR RULE WAS A DISASTER!")
elif percent_of_population_that_starved > 10 or sustainability  < 9:
    print("YOUR RULE WAS MEDIOCRE.")
elif percent_of_population_that_starved > 3 or sustainability  < 10:
    print("YOUR RULE WAS FAIRLY GOOD.")
else:
    print("A FANTASTIC PERFORMANCE!!!")
    
print("SO LONG FOR NOW.")
