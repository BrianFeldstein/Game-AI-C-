#ifndef PLAYVSAI_H_
#define PLAYVSAI_H_

#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include <stdio.h>
#include <eigen3\Eigen\Dense>
#include <vector>
#include "Player.h"
#include "Game.h"

/*
The play_vs_ai function is for Tic-Tac-Toe only.. so really should be moved into the TicTacToe class..
Each game class should really have a way to play against an AI.  It's on the to do list..
*/

Eigen::ArrayXXf readArray(std::string filename)
{
    int rows  = 0;
	std::vector<float> buff(0);
    std::ifstream infile;
    infile.open(filename);
    
	while (! infile.eof())
    {
        std::string line;
        getline(infile, line);
        std::stringstream stream(line);
		rows++;
        float x;
		while(stream >> x)
		{
			buff.push_back(x);
		}
    }
    infile.close();
	int cols = buff.size()/rows;
    Eigen::ArrayXXf result = (Eigen::Map<Eigen::ArrayXXf>(buff.data(),cols,rows)).transpose();
	
    return result;
}


void play_vs_ai(Game * pgame, Player * pplayer, int AI_player_num)
{
	int input;
	Eigen::ArrayXXi state = pgame->state0;
	int player_togo = 1;
	bool cont = true;

	pgame->print_board(state);

	while (cont)
	{

		if (player_togo != AI_player_num)
		{
			std::cout << "Choose a square to play.. 0 to " << state.size()/2 - 1 << std::endl;
			std::cin >> input;
			if (AI_player_num == -1)
			{
				if (state(0,input) != 0) 
				{
					std::cout << "Square already taken.  You suck and lose pathetically." << std::endl;
					pgame->print_board(state); 
					return;
				}
				state(0, input) = 1;
			}
			else
			{
				if (state(0,input + state.size()/2) != 0) 
				{
					std::cout << "Square already taken.  You suck and lose pathetically." << std::endl;
					pgame->print_board(state); 
					return;
				}
				state(0, input + state.size()/2 ) = 1;
			}
		}
		else
		{
			Eigen::ArrayXXi new_state = std::get<0>(pplayer->best_move(pgame, state, false, 1));
			state = new_state;
		}

		if ( std::get<0>(pgame->is_over(state)) )
		{
			std::cout << "Game Over. The Result is " <<  std::get<1>(pgame->is_over(state)) << std::endl;
			pgame->print_board(state);
			return;
		}
		pgame->print_board(state);
		player_togo *= -1;
 	
	}

}

#endif