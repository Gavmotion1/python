p1_score = 0
p2_score = 0
Round = 0

def rps():
    global Round, p1_score, p2_score  
    
    Round += 1

    print(f"Round {Round}")
    print(f"Player 1 Score: {p1_score}")
    print(f"Player 2 Score: {p2_score}")

def get_move(player):
    while True:
        try:
            move = input(f"{player}, enter your move (rock, paper, or scissors): ").lower()
            if move not in ["rock", "paper", "scissors"]:
                raise ValueError("Invalid move. Please choose rock, paper, or scissors.")
            return move
        except ValueError as e:
            print(e)

def play_round():
    global p1_score, p2_score
    
    p1_move = get_move("Player 1")
    p2_move = get_move("Player 2")

    print(f"Player 1 chose: {p1_move}")
    print(f"Player 2 chose: {p2_move}")

    if p1_move == p2_move:
        print('Draw!')
    elif (p1_move == 'rock' and p2_move == 'scissors') or (p1_move == 'scissors' and p2_move == 'paper') or (p1_move == 'paper' and p2_move == 'rock'):
        print('Player 1 won!')
        p1_score += 1
    else:
        print('Player 2 won!')
        p2_score += 1
    
    rps()

while True:
    play_round()
    cont = input("Do you want to play another round? (yes/no): ").lower()
    if cont != 'yes':
        break
