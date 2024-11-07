name_number = [
    ['bob',15],
    ['avery',19],
    ['joe',21],
    ['gabe',54],
    ['talen',90],
    ['conner',45],
    ['jennifer',62],
    ['grog',33],
    ['sam',4],
    ['bertrude',8],
]
greeting_template = 'Welcome to the team, {}! Your team number is {}'
for name,number in name_number:
    greeting = greeting_template.format(name,number)
    print(greeting)
    
