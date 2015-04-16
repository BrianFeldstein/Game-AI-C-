#ifndef PLAYER_H_
#define PLAYER_H_

#include <tuple>
#include <eigen3\Eigen\Dense>
#include <omp.h>
#include <stdio.h>
#include <vector>
#include "Game.h"
//#include "MoveTree.h"

//class MoveTree;

class Player
{
public:
	int player_me;
	Eigen::ArrayXXf theta0;
	Eigen::ArrayXXf theta1;
	Player(int p)
		: player_me(p) {}

	Eigen::ArrayXXf sigmoid(Eigen::ArrayXXf z)
	{
		return (1 + (-z).exp()).inverse();
	}

	std::tuple<float, Eigen::ArrayXXf, Eigen::ArrayXXf> value_dv(Game * pgame, Eigen::ArrayXXi state, int player_togo)
	{	//clock_t t;					

		if (  std::get<0>(pgame->is_over(state))  )
		{
			if (  std::get<1>(pgame->is_over(state)) == 0 ) return std::make_tuple(.5, 0 * theta0, 0 * theta1);
			if (  player_me == std::get<1>(pgame->is_over(state)) ) return std::make_tuple(1, 0 * theta0, 0 * theta1);
			if (  player_me == - std::get<1>(pgame->is_over(state)) ) return std::make_tuple(0, 0 * theta0, 0 * theta1);
		}
	
		Eigen::ArrayXXi stateP(1, 1 + state.cols());
		stateP << Eigen::ArrayXXi::Constant(1,1,1), state;
		Eigen::ArrayXXf z1 = sigmoid(stateP.cast<float>().matrix() * theta0.matrix());
		Eigen::ArrayXXf z1P(1, 1 + z1.cols());
		z1P << Eigen::ArrayXXf::Constant(1,1,1), z1;
		float z2 = sigmoid(z1P.matrix() * theta1.matrix())(0,0);
		Eigen::MatrixXf DVD0;
		Eigen::ArrayXXf DVD1;

		DVD0.noalias() = (stateP.cast<float>().transpose().matrix() * (theta1.block(1,0,theta1.rows()-1,theta1.cols()).transpose() * (z1 - pow(z1,2))).matrix()).operator*((z2 - pow(z2,2)));
		//t = clock();	
		//t = clock()-t;
		//cout << " " << t << endl;
		DVD1 = (z2 - pow(z2,2))*z1P.transpose();

		return std::make_tuple(z2, DVD0.array(), DVD1);
	}

	std::tuple<Eigen::ArrayXXi, bool> best_move(Game * pgame, Eigen::ArrayXXi state, bool rando, float gamefrac)
	{
		std::vector<Eigen::ArrayXXi> st_lst = pgame->state_list(state, player_me);
		std::vector<float> value_list(st_lst.size());

#pragma omp parallel for
		for (int j=0; j < value_list.size(); j++)
		{
			value_list[j] = std::get<0>(value_dv(pgame, st_lst[j], player_me));
		}
	
		int max_ind = distance(value_list.begin(), max_element(value_list.begin(), value_list.end()));
		Eigen::ArrayXXi next_best = st_lst[max_ind];	
		Eigen::ArrayXXi next_state;
	
		if (rando)
		{
			float x = float(rand())/RAND_MAX;
			if (x > 1 - gamefrac)
			{
				next_state = next_best;
			}
			else
			{
				int rand_ind = rand() % st_lst.size();
				next_state = st_lst[rand_ind];
			}
		}
		else
		{
			next_state = next_best;
		}
		bool is_best = (next_state - next_best).isZero();
		return std::make_tuple(next_state, is_best);
	}

};

#endif