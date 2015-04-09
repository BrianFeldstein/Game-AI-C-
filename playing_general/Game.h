#ifndef GAME_H_
#define GAME_H_

#include <eigen3\Eigen\Dense>
#include <vector>
#include <tuple>

class Player;

class Game
{
public:
	Eigen::ArrayXXi state0;
	Game(Eigen::ArrayXXi st) {state0 = st;}

	virtual std::vector<Eigen::ArrayXXi> state_list(Eigen::ArrayXXi state, int player) = 0;
	virtual std::tuple<bool, int> is_over(Eigen::ArrayXXi state) = 0;
	virtual void print_board(Eigen::ArrayXXi state) = 0;
	virtual void play_vs_ai(Player * pplayer, int AI_player_num, int lvl) = 0;
};



#endif