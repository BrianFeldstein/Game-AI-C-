/*
This code trains two AI players to play against each other in a given game.  Currently the game
must be a game of complete information, with no randomness.  Each player uses a neural network
to evaluate the board state.  As they play each other, the network weights are updated via
"reinforcement learning".  The principle behind this is simple:  If an AI thinks that the next state
has a value different than the current state, then this is considered a "mistake", 
and the weights are updated to move the value of the current state towards the value of the next state.
The values of the states in which the game has ended are kept fixed: 1 for a win, 0 for a loss, and
.5 for a tie.

At present, the games implemented are a) Connect Four and b) Tic-Tac-Toe on a rectangular board of any
size, with an arbitrary number "in-a-row" required to win.  I have observed the following values leading
to good training results:  For Connect Four, 1 million games with eps = .1, requiring about 1 day of
training.  For Tic-Tac-Toe on a 3x3 board, learning rate eps = .5, with >150,000 games played.
For a 4x4 board, eps = .1 with 150,000 games played.  On my laptop, this takes about 2 minutes for a 
3x3 board, and 1 hour for a 4x4 board.  Longer training times than these lead to stronger play.

During training, the AI generally chooses for its move the next state with the highest value.  This is 
intentionally not always true, however: a move has a chance to instead be chosen randomly.  The probability
for this starts at 1, and decreases linearly to 0 as training progresses.  This is done so that the
training adequately probes the full set of possible moves to find the best ones.

After the network is trained, when the AI plays against a human player, it then plays by looking through
the game tree to a chosen depth, choosing an action which will maximize the value of the state to 
be obtained, assuming that the opponent is also playing to win.  This thus combines knowledge
from the neural network with a more brute force game tree search, leading to stronger play.

The Game class contains the game rules.  These consist of
1) An initial state, represented by an Eigen array of integers
2) A rule which determines the legal moves, represented as a function taking a state
to a vector of other states.
3) A rule for deciding whether the game has ended, and if so, who has won.
4) Some code to allow for play against a human opponent.

The Player class contains the player AIs.  These consist of 
1) The neural network weights ("thetas").
2) A rule for determining the value of the board (as well as the derivatives of the value
with respect to the thetas)
3) A rule for choosing the next move, either based on the state with the highest value, or
randomly as appropriate.

The Move_Tree class sets up the game tree for a given board position, and determines the best 
move given an input search depth ("level").
*/

#include <iostream>
#include <string>
#include <tuple>
#include <ctime>
#include <cmath>
#include <vector>
#include <eigen3\Eigen\Dense>
#include <omp.h>
#include <stdio.h>
#include <fstream>
#include "Game.h"
#include "Player.h"
#include "TicTacToe.h"
#include "ConnectFour.h"
#include "ReadArray.h"

using namespace Eigen;
using namespace std;

/*
This function takes a pointer to a game object, as well as pointers to 2 players, and updates the player
AI's via reinforcement learning.
"learner" specifies which players will have their network weights updated
(as opposed to just choosing the best moves without learning.. 1 for p1, -1 for p2, 0 for both).
If you want no one to learn, set learner to an int other than -1,0,1.
"gamefrac" is the fraction of total training games we have finished.  This will determine the chance to move
randomly.
"test" is currently not implemented.
if "pr" is true, the game will have all its moves written out. 
"eps" is the learning rate, and "gamma" is a factor which causes learning to be faster for moves that
are later in a given game.  It seems fine to set this to 1 so that it doesn't do anything.
*/
tuple<int, float> play_game(Game *pgame, Player *pplayer1, Player * pplayer2, int learner, float gamefrac, bool test, bool pr, float gamma, float eps)
{
	ArrayXXi state = pgame->state0;

	int player_togo = 1;
	int TotTurns = 0;
	int TotDeltaV = 0; //not currently used
	bool cont = true;
	
	ArrayXXi new_state;
	tuple<ArrayXXi, bool> b_m;
	bool is_best = false;
	
	tuple<float, ArrayXXf, ArrayXXf> vdv_1_state;
	tuple<float, ArrayXXf, ArrayXXf> vdv_1_nstate;
	ArrayXXf theta0p1temp;
	ArrayXXf theta1p1temp;
	tuple<float, ArrayXXf, ArrayXXf> vdv_2_state;
	tuple<float, ArrayXXf, ArrayXXf> vdv_2_nstate;
	ArrayXXf theta0p2temp;
	ArrayXXf theta1p2temp;
	
	while (cont)
	{ 		
		if (player_togo == pplayer1->player_me) b_m = pplayer1->best_move(pgame, state, player_togo==learner||learner==0, gamefrac);
		if (player_togo == pplayer2->player_me) b_m = pplayer2->best_move(pgame, state, player_togo==learner||learner==0, gamefrac);
		new_state = get<0>(b_m);
		is_best = true;//get<1>(b_m);

		//Something from the python version I might implement later, but isn't really important:
		//if test and TotTurns in [0,1,2,3,4,5,6,7,8]:
        //    Ans = GetAvgXRes(State,50)
        //    TotDeltaV += np.abs(ValueDV(State, Theta0X, Theta1X, 1)[0] - Ans)

		TotTurns += 1;
		
#pragma omp parallel sections 
		{
		{
		if( (learner == 1 || learner == 0) && is_best && TotTurns >= 1)
		{	
			vdv_1_state = pplayer1->value_dv(pgame, state, player_togo);
			vdv_1_nstate = pplayer1->value_dv(pgame, new_state, player_togo);
			//cout << "old " << get<0>(vdv_1_state) << endl;
			//cout << "next " << get<0>(vdv_1_nstate) << endl;
			theta0p1temp = pplayer1->theta0 + pow(gamma, -TotTurns) * eps * (get<0>(vdv_1_nstate) - get<0>(vdv_1_state)) * get<1>(vdv_1_state);
			theta1p1temp = pplayer1->theta1 + pow(gamma, -TotTurns) * eps * (get<0>(vdv_1_nstate) - get<0>(vdv_1_state)) * get<2>(vdv_1_state);
			pplayer1->theta0 = theta0p1temp;
			pplayer1->theta1 = theta1p1temp;
			//cout << "new " << get<0>(pplayer1->value_dv(pgame, state, player_togo)) << endl;
		}
		}
		#pragma omp section
		{
		if( (learner == -1 || learner == 0) && is_best && TotTurns >= 1)
		{	
			vdv_2_state = pplayer2->value_dv(pgame, state, player_togo);
			vdv_2_nstate = pplayer2->value_dv(pgame, new_state, player_togo);
			theta0p2temp = pplayer2->theta0 + pow(gamma, -TotTurns) * eps * (get<0>(vdv_2_nstate) - get<0>(vdv_2_state)) * get<1>(vdv_2_state);
			theta1p2temp = pplayer2->theta1 + pow(gamma, -TotTurns) * eps * (get<0>(vdv_2_nstate) - get<0>(vdv_2_state)) * get<2>(vdv_2_state);
			pplayer2->theta0 = theta0p2temp;
			pplayer2->theta1 = theta1p2temp;
		}
		}
		}
		if (pr) cout << "hi" << get<0>(pplayer1->value_dv(pgame, new_state, player_togo)) << " , " << get<0>(pplayer2->value_dv(pgame, new_state, player_togo)) << endl;
		
		cont = ! get<0>(pgame->is_over(new_state));
		state = new_state;
		
		if (pr) pgame->print_board(state);

		player_togo = -1*player_togo;
	}
	//if (get<1>(is_over(gprop, new_state)) == 1 ) cout << new_state << endl; 
	return make_tuple(get<1>(pgame->is_over(new_state)), TotDeltaV/TotTurns);
}

/*
This function causes a chosen number of games "game_num" to be played by two players.
*/
tuple<vector<float>, vector<float>> learn(Game *pgame, Player *pplayer1, Player *pplayer2, int game_num, float gamma, float eps)
{	

	tuple<int, float> pg;
	
	cout << "go" << endl;
	int learner = 0;
	vector<float> ties_list(0);
	vector<float> p1Wins_list(0);
	vector<float> avdv_list(0);
	int num_ties = 0;
	int num_p1Wins = 0;
	float temp_avdv = 0;
	
	for (int i = 0; i < game_num; i++)
	{		
		//cout << i << endl;
		bool dotest = false;
		if (i % 10000 < 1000) dotest = false;
		if (i%1000 == 0)
		{
			cout << "(" << i <<" , " << learner << ")" << endl;
			pg = play_game(pgame, pplayer1, pplayer2, learner, float(i)/game_num, dotest, true, gamma, eps);
			int res = get<0>(pg);
			float avdv = get<1>(pg);
			temp_avdv += avdv;
			if (res == 0 ) num_ties += 1;
			if (res == 1 ) num_p1Wins += 1;
			avdv_list.push_back(temp_avdv/float(1000));
			ties_list.push_back(num_ties/float(1000));
			p1Wins_list.push_back(num_p1Wins/float(1000));
			cout << num_ties/float(1000) << " , " << num_p1Wins/float(1000) << endl;
			temp_avdv = 0;
			num_ties = 0;
			num_p1Wins = 0;
			//plot ties list?		
		}
		else
		{
			pg = play_game(pgame, pplayer1, pplayer2, learner, float(i)/game_num, dotest, false, gamma, eps);
			int res = get<0>(pg);
			float avdv = get<1>(pg);
			temp_avdv += avdv;
			if (res == 0) num_ties += 1;
			if (res == 1 ) num_p1Wins += 1;
			//cout << res << endl;
		
		}
	}
	return make_tuple(ties_list, p1Wins_list);
}

	
/*
"main" sets up the game, initializes the player AI's, and and sets them playing against each other.
After play has finished, the neural networks weights are saved to text files.
*/
int main()
{
	srand(time(0));
	srand(rand());
	
	bool train = false;
	bool playAI = true;
	Game * pgame;

	int game_val;
	string folder;

	cout << "Choose Game:" << endl;
	cout << "0: 3x3 Tic Tac Toe"<< endl;
	cout << "1: 4x4 Tic Tac Toe"<< endl;
	cout << "2: Connect Four"<< endl;
	cin >> game_val;
		
	if (game_val == 0)
	{
		//TicTacToe TTTGame(3,3,3);
		pgame = new TicTacToe(3,3,3);//TTTGame;
		folder = "Theta3332000000p5/";
	}
	else if (game_val == 1)
	{
		//TicTacToe TTTGame(4,4,4);
		pgame = new TicTacToe(4,4,4);//&TTTGame;
		folder = "Theta44415000000p1/";
	}
	else if (game_val == 2)
	{
		//ConnectFour CFGame(6,7,4);	
		pgame = new ConnectFour(6,7,4);//&CFGame;
		folder = "ThetaC41000000p1/";
	}


	if (train)
	{
		int state_len = pgame->state0.size();
		int hidden = 3*state_len;

		Player player1(1);
		Player player2(-1);
		player1.theta0 = .5 * ArrayXXf::Random(state_len + 1, hidden);
		player1.theta1 = .5 * ArrayXXf::Random(hidden + 1, 1);
		player2.theta0 = .5 * ArrayXXf::Random(state_len + 1, hidden);
		player2.theta1 = .5 * ArrayXXf::Random(hidden + 1, 1);

		Player *pplayer1 = &player1;
		Player *pplayer2 = &player2;

		float gamma = 1; float eps = .1;
		int NumGames = 1;
		tuple<vector<float>, vector<float>> Data = learn(pgame, pplayer1, pplayer2, NumGames, gamma, eps);

		ofstream x0file;
		x0file.open("theta0p1.txt");
		x0file << player1.theta0;
		x0file.close();

		ofstream x1file;
		x1file.open("theta1p1.txt");
		x1file << player1.theta1;
		x1file.close();

		ofstream o0file;
		o0file.open("theta0p2.txt");
		o0file << player2.theta0;
		o0file.close();

		ofstream o1file;
		o1file.open("theta1p2.txt");
		o1file << player2.theta1;
		o1file.close();
	}

	if (playAI)
	{
		int AI_player_num;
		int lvl;

		cout << "Which Player Will Be The AI?" << endl;
		cout<< "1: Player 1" << endl;
		cout<< "-1: Player 2" << endl;
		cin >> AI_player_num;

		cout << "Enter The AI Difficulty Level.. 0-4." << endl;
		cin >> lvl;
		
		//string folder = "ThetaC41000000p1/";//"./";//"Theta44415000000p1/";
		//int AI_player_num = 1;
		//int lvl = 3;
		Player AIplayer(AI_player_num);
		Player *pplayer = &AIplayer;
		if (AIplayer.player_me == 1)
		{
			AIplayer.theta0 = readArray(folder + "theta0p1.txt");
			AIplayer.theta1 = readArray(folder + "theta1p1.txt");
		}
		if (AIplayer.player_me == -1)
		{
				AIplayer.theta0 = readArray(folder + "theta0p2.txt");
				AIplayer.theta1 = readArray(folder + "theta1p2.txt");
		}
		pgame->play_vs_ai(pplayer, AI_player_num, lvl);
	}

	string YorN;
	cout << "Play Again? Y or N" << endl;
	cin >> YorN;
	if (YorN == "Y") return main();
	else return 0;
}