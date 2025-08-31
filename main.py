import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

# --- Helper functions for board and move representation ---

def board_to_tensor(board: chess.Board):
    """
    Converts a chess.Board object into a 12x8x8 PyTorch tensor.
    The tensor represents the board state from the current player's perspective.
    
    Channels: 
    0-5: White pieces (pawn, knight, bishop, rook, queen, king)
    6-11: Black pieces (pawn, knight, bishop, rook, queen, king)
    """
    board_state = torch.zeros(12, 8, 8, dtype=torch.float32)
    piece_map = board.piece_map()
    
    # Map piece types to channel indices
    piece_to_channel = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    for square, piece in piece_map.items():
        row, col = chess.square_file(square), chess.square_rank(square)
        channel = piece_to_channel[piece.piece_type]
        
        if piece.color == chess.WHITE:
            board_state[channel, row, col] = 1.0
        else:
            board_state[channel + 6, row, col] = 1.0
            
    # For the model's perspective, we always orient the board for the current player
    if board.turn == chess.BLACK:
        board_state = board_state.flip(1, 2) # Flip both row and col dimensions

    return board_state.unsqueeze(0) # Add batch dimension

def move_to_index(move: chess.Move):
    """
    Maps a chess.Move object to a single integer index (0-4671).
    This mapping is based on the AlphaZero paper's move representation:
    - 64x64 for normal moves (4096)
    - 56 promotions (7 types x 8 possible squares)
    - 8 underpromotions to Knight (4 squares x 2 types)
    - Castling moves
    """
    from_sq = move.from_square
    to_sq = move.to_square

    # Normal moves
    if not move.promotion:
        return from_sq * 64 + to_sq

    # Promotions
    promotions = [
        chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT
    ]
    promo_piece = move.promotion
    
    # Underpromotions (moves that are not queen promotions)
    if promo_piece != chess.QUEEN:
        from_rank = chess.square_rank(from_sq)
        to_rank = chess.square_rank(to_sq)
        
        # Determine the promotion index based on the "to" square
        if promo_piece == chess.ROOK:
            promo_index = 0
        elif promo_piece == chess.BISHOP:
            promo_index = 1
        else: # Knight
            promo_index = 2
            
        return 4096 + to_sq + promo_index * 64

    # The mapping here is a simplified approach, a full AlphaZero
    # mapping is more complex. Let's use a simpler mapping for now.
    # Total moves = 64*64 (4096) + Promotions
    # Let's map promotions as a separate block
    
    promo_index = promotions.index(promo_piece)
    # A simplified but functional mapping for demonstration
    return 4096 + (promo_index * 64) + to_sq

# We also need a function to reverse this for self-play
def index_to_move(board: chess.Board, index: int):
    if index < 4096:
        from_sq = index // 64
        to_sq = index % 64
        move = chess.Move(from_sq, to_sq)
        if move in board.legal_moves:
            return move
    
    # Handle promotions (simplified)
    elif index >= 4096:
        # This is a very simplistic mapping and needs more logic
        # For this example, we'll assume a direct mapping for simplicity
        # The true AlphaZero model has a more complex policy head
        pass
    
    return None # Return None if the move is not found

# --- Neural Network Model ---

class ChessModel(nn.Module):
    """
    Neural network model for chess, inspired by AlphaZero.
    It has two heads: a policy head for move probabilities and a value head for position evaluation.
    """
    def __init__(self):
        super(ChessModel, self).__init__()

        # CNN block: input 12 channels (6 pieces x 2 colors), board 8x8
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Policy head (where to move)
        self.policy_conv = nn.Conv2d(128, 32, kernel_size=1)
        # 4672 is a standard move representation in AlphaZero
        self.policy_fc = nn.Linear(32 * 8 * 8, 4672)  

        # Value head (position evaluation)
        self.value_conv = nn.Conv2d(128, 16, kernel_size=1)
        self.value_fc1 = nn.Linear(16 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # CNN feature extractor
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Policy head
        p = F.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1) # Flatten for linear layer
        p = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1) # Flatten for linear layer
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # value from -1 to 1

        return p, v

# --- Training and Self-Play Logic ---

def get_move_reward(board: chess.Board, move: chess.Move):
    """
    Calculates a reward for a single move, primarily for captures.
    """
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        return piece_values.get(captured.piece_type, 0)
    return 0

def self_play_game(model: ChessModel):
    """
    Simulates a full chess game using the model to choose moves.
    Returns the game states, moves, and rewards.
    """
    board = chess.Board()
    states = []
    moves = []
    rewards = []

    while not board.is_game_over():
        x = board_to_tensor(board)
        policy_output, value_output = model(x)

        legal_moves = list(board.legal_moves)
        
        # Эпсилон-жадный подход: 10% шанс на случайный ход
        if random.random() < 0.1:
            best_move = random.choice(legal_moves)
        else:
            legal_indices = [move_to_index(m) for m in legal_moves]
            policy_output_legal = policy_output[0, legal_indices]
            best_index_in_legal_list = torch.argmax(policy_output_legal).item()
            best_move = legal_moves[best_index_in_legal_list]
        
        # Find the global index of the best move
        best_move_global_index = move_to_index(best_move)

        # Save state and move
        states.append(x)
        moves.append(best_move_global_index)

        # Calculate intermediate reward for captures
        rewards.append(get_move_reward(board, best_move))

        board.push(best_move)
    print(board)

    # Final game result
    outcome = board.result()
    if outcome == "1-0":
        final_reward = 1
    elif outcome == "0-1":
        final_reward = -1
    else:
        final_reward = 0

    # Add the final result to all intermediate rewards
    rewards = [r + final_reward for r in rewards]

    return states, moves, rewards

def train_on_game(model, states, moves, rewards):
    """
    Trains the model on a single game's data.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for x, move_idx, reward in zip(states, moves, rewards):
        # Pass the state through the model
        policy_output, value_output = model(x)

        # Policy loss: cross-entropy loss for move prediction
        # The target is a tensor with 1.0 at the correct move index
        target_policy = torch.zeros_like(policy_output).squeeze(0)
        target_policy[move_idx] = 1
        policy_loss = F.cross_entropy(policy_output, target_policy.unsqueeze(0))

        # Value loss: mean squared error for position evaluation
        target_value = torch.tensor([[reward]], dtype=torch.float32)
        value_loss = F.mse_loss(value_output, target_value)

        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# --- Main training loop ---

if __name__ == '__main__':
    model = ChessModel()
    print("Starting training...")
    
    # You can uncomment this line if you have a GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    for episode in range(10000):  # 100 games
        try:
            states, moves, rewards = self_play_game(model)
            train_on_game(model, states, moves, rewards)
            
            # Final game result is the last reward
            final_reward = rewards[-1]
            if final_reward > 0:
                result = "White wins" if final_reward == 1 else "Black wins"
            elif final_reward < 0:
                result = "Black wins" if final_reward == -1 else "White wins"
            else:
                result = "Draw"
            
            print(f"Game {episode} finished. Final game outcome: {result}. Last reward: {final_reward}")
        except Exception as e:
            print(f"An error occurred during game {episode}: {e}")
            continue

    print("Training complete.")
