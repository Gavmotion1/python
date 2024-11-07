import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import defaultdict

# Neural Network Model
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Output a single value for board evaluation

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # Output in range -1 to 1

# Initialize the model, optimizer, and loss function
model = ChessNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Convert board to tensor
def board_to_tensor(board):
    piece_map = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 200,
                 'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': -200}
    board_tensor = np.zeros(64, dtype=np.float32)
    for i, piece in enumerate(board.board_fen().replace("/", "")):
        if piece.isdigit():
            i += int(piece) - 1
        elif piece != '.':
            board_tensor[i] = piece_map[piece]
    return torch.tensor(board_tensor).float().unsqueeze(0)

# MCTS Node
class MCTSNode:
    def __init__(self, board, parent=None):
        self.board = board
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0

    def is_fully_expanded(self):
        return len(self.children) == len(list(self.board.legal_moves))

    def add_child(self, move):
        board_copy = self.board.copy()
        board_copy.push(move)
        child = MCTSNode(board_copy, self)
        self.children.append(child)
        return child

# MCTS Functions
def select(node):
    best_value = -float('inf')
    best_node = None
    for child in node.children:
        uct_value = (child.wins / (child.visits + 1)) + 1.41 * np.sqrt(np.log(node.visits + 1) / (child.visits + 1))
        if uct_value > best_value:
            best_value = uct_value
            best_node = child
    return best_node

def expand(node):
    untried_moves = [move for move in node.board.legal_moves if move not in [child.board.peek() for child in node.children]]
    if not untried_moves:
        return None
    move = random.choice(untried_moves)
    return node.add_child(move)

def simulate(node):
    board_tensor = board_to_tensor(node.board)
    with torch.no_grad():
        return model(board_tensor).item()

def backpropagate(node, result):
    while node:
        node.visits += 1
        node.wins += result
        node = node.parent

def mcts(board, simulations=100):
    root = MCTSNode(board)
    for _ in range(simulations):
        node = root
        # Selection
        while node.is_fully_expanded() and node.children:
            node = select(node)
        # Expansion
        if not node.is_fully_expanded():
            node = expand(node)
        if node is None:
            continue
        # Simulation
        result = simulate(node)
        # Backpropagation
        backpropagate(node, result)
    return max(root.children, key=lambda child: child.visits).board.peek()  # Return the most visited move

# Training Loop (Self-Play for Reinforcement Learning)
def train(model, optimizer, criterion, epochs=10):
    for epoch in range(epochs):
        board = chess.Board()
        while not board.is_game_over():
            move = mcts(board)
            board.push(move)
            board_tensor = board_to_tensor(board)
            score = model(board_tensor)
            target = torch.tensor([[1]]) if board.is_checkmate() else torch.tensor([[0]])
            optimizer.zero_grad()
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} completed")

# Start Training
train(model, optimizer, criterion, epochs=10)
