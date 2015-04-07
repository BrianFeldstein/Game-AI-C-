#ifndef MOVETREE_H_
#define MOVETREE_H_

#include <vector>
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
	float current_val;
	float biggest_val;

	move_tree(node * t, int lvls)
		: top(t), levels(lvls)
	{
		maxvalind = 0;
		biggest_val = 0;
		setup_node(top, levels);

		for(int i = 0; i < top->children.size(); i++)
		{	
			current_val = val_from_future(top->children[i], levels);
			if (current_val > biggest_val) {biggest_val = current_val; maxvalind = i;}
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

	float val_from_future(node * t0, int l)
	{	
		std::vector<float> vals;
		if (l == 0 || std::get<0>( t0->pgame->is_over(t0->state))) {return t0->val;}
		l--;
		for (int i=0; i< t0->children.size(); i++)
		{
			vals.push_back(val_from_future(t0->children[i], l));
		}
		
		if (t0->player_togo == top->player_togo) return *std::max_element(std::begin(vals), std::end(vals));
		else if (t0->player_togo == -1* top->player_togo) return *std::min_element(std::begin(vals), std::end(vals));

	}

};


#endif