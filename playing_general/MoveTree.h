#ifndef MOVETREE_H_
#define MOVETREE_H_

#include <vector>
#include <tuple>
#include <eigen3\Eigen\Dense>
#include "Game.h"
#include "Player.h"


class node
{
public:
	Eigen::ArrayXXi state;
	Game * pgame;
	Player * pplayer;
	int player_togo;
	float val;
	std::vector<node *> children;
	node(Game * g, Player * p, Eigen::ArrayXXi st, int ptg)
		: player_togo(ptg), state(st), pgame(g), pplayer(p), val(std::get<0>(p->value_dv(pgame, state, player_togo))) {}
	~node() {for (int i=0;i<children.size();i++) {delete children[i];} }
};

class move_tree
{
public:
	node * top;
	int levels;
	Eigen::ArrayXXi next_state;
	std::vector<float> vals_future_list;
	int maxvalind;
	std::tuple<float, int>current_res;
	float current_val;
	int current_ldiff;
	float biggest_val;
	int biggest_ldiff;

	move_tree(node * t, int lvls)
		: top(t), levels(lvls)
	{
		maxvalind = 0;
		biggest_val = 0;
		biggest_ldiff = -1;
		setup_node(top, levels);

		for(int i = 0; i < top->children.size(); i++)
		{	
			current_res = val_from_future(top->children[i], levels);
			current_val = std::get<0>(current_res);
			current_ldiff = std::get<1>(current_res);
			//std:: cout << current_val << " " << current_ldiff << std::endl;
			if (current_val >= biggest_val-.000001 && current_ldiff >= biggest_ldiff) {biggest_val = current_val; biggest_ldiff = current_ldiff; maxvalind = i;}
		}
		next_state = top->children[maxvalind]->state;
	}


	void setup_node(node * t0, int l)
	{
		std::vector<Eigen::ArrayXXi> child_state_list; 

		if (l < 0 || std::get<0>( t0->pgame->is_over(t0->state))) return;
		else child_state_list = t0->pgame->state_list(t0->state, t0->player_togo);
		for (int i = 0; i < child_state_list.size(); i++)
		{	
			node * temp = new node(t0->pgame, t0->pplayer, child_state_list[i], -1*t0->player_togo);
			t0->children.push_back(temp);
		}

		for (int i = 0; i < child_state_list.size(); i++)
		{
			setup_node(t0->children[i], l-1);
		}
	}

	std::tuple<float, int> val_from_future(node * t0, int l)
	{	
		int best_index;
		std::tuple<float, int> res;
		std::vector<float> vals;
		std::vector<int> ldiffvals;
		std::vector<float> combined_vals;
		if (l == 0 || std::get<0>( t0->pgame->is_over(t0->state))) {return std::make_tuple(t0->val, levels-l);}
		l--;
		for (int i=0; i< t0->children.size(); i++)
		{
			res = val_from_future(t0->children[i], l);
			vals.push_back(std::get<0>(res));
			ldiffvals.push_back(std::get<1>(res));
			combined_vals.push_back( std::get<0>(res)* std::pow(10.0,5) + std::get<1>(res) );
		}
		
		if (t0->player_togo == top->player_togo)
		{
			best_index = std::distance(combined_vals.begin(), std::max_element(combined_vals.begin(), combined_vals.end()));
			return std::make_tuple(vals[best_index], ldiffvals[best_index]);
		}
		else if	(t0->player_togo == -1* top->player_togo)
		{	
			best_index = std::distance(combined_vals.begin(), std::min_element(combined_vals.begin(), combined_vals.end()));
			return std::make_tuple(vals[best_index], ldiffvals[best_index]);
		}
	}

};


#endif