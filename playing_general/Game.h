#ifndef GAME_H_
#define GAME_H_

#include <eigen3\Eigen\Dense>
#include <vector>
#include <tuple>

class Game
{
public:
	Eigen::ArrayXXi state0;
	Game(Eigen::ArrayXXi st) {state0 = st;}

	virtual std::vector<Eigen::ArrayXXi> state_list(Eigen::ArrayXXi state, int player) = 0;
	virtual std::tuple<bool, int> is_over(Eigen::ArrayXXi state) = 0;
	virtual void print_board(Eigen::ArrayXXi state) = 0;
};

class TicTacToe: public Game
{
private:
	Eigen::ArrayXXi convert(Eigen::ArrayXXi stRC)
	{
		Eigen::ArrayXXi st2RC = Eigen::ArrayXXi::Zero(1,2*stRC.cols());
		for(int i=0; i<stRC.cols(); i++)
		{
			if (stRC(0,i) == 1) st2RC(0,i) = 1;
			if (stRC(0,i) == -1) st2RC(0, i + stRC.cols()) = 1;
		}
		return st2RC;
	}



public:
	int R; int C; int W; //rows and columns of board, and number in-a-row required to win.

	TicTacToe(int R0, int C0, int W0)
		: Game(Eigen::ArrayXXi::Zero(1,2*R0*C0))
		, R(R0), C(C0), W(W0)
		{}

		
	std::vector<Eigen::ArrayXXi> state_list(Eigen::ArrayXXi state, int player)
	{
		Eigen::ArrayXXi stateRC = state.block(0,0,1,R*C) - state.block(0,R*C,1,R*C);
			
		std:: vector<Eigen::ArrayXXi> st_list(0);
		Eigen::ArrayXXi temp_stateRC;
		Eigen::ArrayXXi temp_state2RC;

		for(int i=0; i < stateRC.cols(); i++)
		{
			if (stateRC(0,i)==0)
			{
				temp_stateRC = stateRC;
				temp_stateRC(0,i) = player;
				temp_state2RC = convert(temp_stateRC);
				st_list.push_back(temp_state2RC);
			}
		}
		return st_list;
	}
/*
This method figures out if the game is over, and who the winner is.  It currently looks through every 
board square, and then looks right, down, and diagonally for lines.  This could be done more efficiently, 
but at the moment this is not the bottleneck for speed, so I'm leaving it.
*/
	std::tuple<bool, int> is_over(Eigen::ArrayXXi state)
	{
		Eigen::ArrayXXi stateRC1(R,C);

		if ( (state.block(0,0,1,R*C)).sum() > (state.block(0,R*C,1,R*C)).sum() )
		{
			stateRC1 = Eigen::Map<Eigen::ArrayXXi>(state.block(0,0,1,R*C).data(),C,R).transpose();
		}
		else
		{
			stateRC1 <<  - Eigen::Map<Eigen::ArrayXXi>(state.block(0,R*C,1,R*C).data(),C,R).transpose();
		}	

		int k; bool still_going;

		for (int i = 0; i < stateRC1.rows(); i++)
		{
			for (int j = 0; j < stateRC1.cols(); j++)
			{
				if (stateRC1(i,j) == 0) continue;
			
				k = 0;
				still_going = true;
				while(still_going && k < W && k+i < stateRC1.rows()  )
				{
					if (stateRC1(i,j) == stateRC1(i+k,j)) k+=1;
					else still_going = false;
					if (k == W) return std::make_tuple(true, stateRC1(i,j));
				}

				k = 0;
				still_going = true;
				while(still_going && k < W && k+j < stateRC1.cols()  )
				{
					if (stateRC1(i,j) == stateRC1(i,j+k)) k+=1;
					else still_going = false;
					if (k == W) return std::make_tuple(true, stateRC1(i,j));
				}

				k = 0;
				still_going = true;
				while(still_going && k < W && k+i < stateRC1.rows() && k+j < stateRC1.cols()   )
				{
					if (stateRC1(i,j) == stateRC1(i+k,j+k)) k+=1;
					else still_going = false;
					if (k == W) return std::make_tuple(true, stateRC1(i,j));
				}

				k = 0;
				still_going = true;
				while(still_going && k < W && -k+i >= 0 && k+j < stateRC1.cols()   )
				{
					if (stateRC1(i,j) == stateRC1(i-k,j+k)) k+=1;
					else still_going = false;
					if (k == W) return std::make_tuple(true, stateRC1(i,j));
				}
			}
		}
		
		if (state.sum() == R*C) return std::make_tuple(true, 0); 
		return std::make_tuple(false, 0);
	}

	void print_board(Eigen::ArrayXXi state)
	{
		Eigen::ArrayXXi stateRC = state.block(0,0,1,R*C) - state.block(0,R*C,1,R*C);
		Eigen::ArrayXXi stateRCrect(R,C);
		stateRCrect = Eigen::Map<Eigen::ArrayXXi>(stateRC.data(),C,R).transpose();
		std::cout << stateRCrect << std::endl;
		std::cout << "" << std::endl;
	}

};

#endif