# GameAICpp
Train and play an AI via  reinforcement learning + a neural network for a general game with specifiable rules.
playing_generalPreOO.cpp is outdated and should not be used.  playing_general.cpp is the correct module.

This code trains two AI players to play against each other in a given game.  Currently the game
must be a game of complete information, with no randomness.  Each player uses a neural network
to evaluate the board state.  As they play each other, the network weights are updated via
"reinforcement learning".  The principle behind this is simple:  If an AI thinks that the next state
has a value different than the current state, then this is considered a "mistake", 
and the weights are updated to move the value of the current state towards the value of the next state.
The values of the states in which the game has ended are kept fixed: 1 for a win, 0 for a loss, and
.5 for a tie.

At present, the game implemented is Tic-Tac-Tie on a rectangular board of any size, with
an arbitrary number "in-a-row" required to win.  I have observed the following values leading to good
training results:  For a 3x3 board, learning rate eps = .5, with >150,000 games played.
For a 4x4 board, eps = .1 with 5,000,000 games played.  On my laptop, this takes about 2 minutes for a 
3x3 board, and 10 hours for a 4x4 board.  The 3x3 numbers are somewhat optimized, the 4x4 numbers can hopefully
come down a lot, I've just quoted what I've found so far.

When trying to win, a player AI always chooses for his move the next state with the highest value.  During
training, however, a move has a chance to instead be chosen randomly.  The probability for this starts
at 1, and decreases linearly to 0 as training progresses.  This is done so that the training adequately
probes the full set of possible moves to find the best ones.

The Game class contains the game rules.  These consist of
1) An initial state, represented by an Eigen array of integers
2) A rule which determines the legal moves, represented as a function taking a state
to a vector of other states.
3) A rule for deciding whether the game has ended, and if so, who has won.

The Player class contains the player AIs.  These consist of 
1) The neural network weights ("thetas").
2) A rule for determining the value of the board (as well as the derivatives of the value
with respect to the thetas)
3) A rule for choosing the next move, either based on the state with the highest value, or
randomly as appropriate.

To Play Against the AI, in "main", set train = false, and playAI = true.  You have to set the folder where the neural
net weights to play against are saved.
