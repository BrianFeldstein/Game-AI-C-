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

using namespace Eigen;
using namespace std;


//gprop = (R,C,W) are the Rows and Columns in the board,  and number in a row to Win.

ArrayXXf sigmoid(ArrayXXf z)
{
	return (1 + (-z).exp()).inverse();
}

ArrayXXi convert(ArrayXXi stRC)
{
	ArrayXXi st2RC = ArrayXXi::Zero(1,2*stRC.cols());
	for(int i=0; i<stRC.cols(); i++)
	{
		if (stRC(0,i) == 1) st2RC(0,i) = 1;
		if (stRC(0,i) == -1) st2RC(0, i + stRC.cols()) = 1;
	}
	return st2RC;
}

tuple<bool, int> is_over(vector<int> gprop, ArrayXXi state2RC)
{
	int R = gprop[0]; int C = gprop[1]; int W = gprop[2];

	ArrayXXi stateRC1(R,C);

	if ( (state2RC.block(0,0,1,R*C)).sum() > (state2RC.block(0,R*C,1,R*C)).sum() )
	{
		stateRC1 = Map<ArrayXXi>(state2RC.block(0,0,1,R*C).data(),R,C);
	}
	else
	{
		stateRC1 <<  - Map<ArrayXXi>(state2RC.block(0,R*C,1,R*C).data(),R,C);
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
				if (k == W) return make_tuple(true, stateRC1(i,j));
			}

			k = 0;
			still_going = true;
			while(still_going && k < W && k+j < stateRC1.cols()  )
			{
				if (stateRC1(i,j) == stateRC1(i,j+k)) k+=1;
				else still_going = false;
				if (k == W) return make_tuple(true, stateRC1(i,j));
			}

			k = 0;
			still_going = true;
			while(still_going && k < W && k+i < stateRC1.rows() && k+j < stateRC1.cols()   )
			{
				if (stateRC1(i,j) == stateRC1(i+k,j+k)) k+=1;
				else still_going = false;
				if (k == W) return make_tuple(true, stateRC1(i,j));
			}

			k = 0;
			still_going = true;
			while(still_going && k < W && -k+i >= 0 && k+j < stateRC1.cols()   )
			{
				if (stateRC1(i,j) == stateRC1(i-k,j+k)) k+=1;
				else still_going = false;
				if (k == W) return make_tuple(true, stateRC1(i,j));
			}
		}
	}
		
	if (state2RC.sum() == R*C) return make_tuple(true, 0); 
	return make_tuple(false, 0);
}


tuple<float, ArrayXXf, ArrayXXf> value_dv(vector<int> gprop, ArrayXXi state2RC, ArrayXXf theta0, ArrayXXf theta1, int player)
{	//clock_t t;					

	if (  get<0>(is_over(gprop, state2RC))  )
	{
		if (  get<1>(is_over(gprop, state2RC)) == 0 ) return make_tuple(.5, 0 * theta0, 0 * theta1);
		if (  player == get<1>(is_over(gprop, state2RC)) ) return make_tuple(1, 0 * theta0, 0 * theta1);
		if (  player == - get<1>(is_over(gprop, state2RC)) ) return make_tuple(0, 0 * theta0, 0 * theta1);
	}
	
	ArrayXXi stateP(1, 1 + state2RC.cols());
	stateP << ArrayXXi::Constant(1,1,1), state2RC;
	ArrayXXf z1 = sigmoid(stateP.cast<float>().matrix() * theta0.matrix());
	ArrayXXf z1P(1, 1 + z1.cols());
	z1P << ArrayXXf::Constant(1,1,1), z1;
	float z2 = sigmoid(z1P.matrix() * theta1.matrix())(0,0);
	MatrixXf DVD0;
	ArrayXXf DVD1;

	DVD0.noalias() = (z2 - pow(z2,2))*(stateP.cast<float>().transpose().matrix() * (theta1.block(1,0,theta1.rows()-1,theta1.cols()).transpose() * (z1 - pow(z1,2))).matrix());
	//t = clock();	
	//t = clock()-t;
	//cout << " " << t << endl;
	DVD1 = (z2 - pow(z2,2))*z1P.transpose();

	//if (state18.isZero()) return make_tuple(.5, 0 * DVD0.array(), 0 * DVD1);
	return make_tuple(z2, DVD0.array(), DVD1);
}

tuple<ArrayXXi, bool> best_move(vector<int> gprop, ArrayXXi state2RC, ArrayXXf theta0, ArrayXXf theta1, int player, bool rando, float gamefrac)
{
	int R = gprop[0]; int C = gprop[1];
	ArrayXXi stateRC = state2RC.block(0,0,1,R*C) - state2RC.block(0,R*C,1,R*C);
	vector<ArrayXXi> state_list(0);
	ArrayXXi temp_stateRC;
	ArrayXXi temp_state2RC;

	for(int i=0; i < stateRC.cols(); i++)
	{
		if (stateRC(0,i)==0)
		{
			temp_stateRC = stateRC;
			temp_stateRC(0,i) = player;
			temp_state2RC = convert(temp_stateRC);
			state_list.push_back(temp_state2RC);
		}
	}
		
	vector<float> value_list(state_list.size());
	
#pragma omp parallel for
	for (int j=0; j < value_list.size(); j++)
	{
		value_list[j] = get<0>(value_dv(gprop, state_list[j], theta0, theta1, player));
	}
	
	int max_ind = distance(value_list.begin(), max_element(value_list.begin(), value_list.end()));
	ArrayXXi next_best = state_list[max_ind];	
	ArrayXXi next_state;
	
	if (rando)
	{
		float x = float(rand())/RAND_MAX;
		if (x > 1 - gamefrac)
		{
			next_state = next_best;
		}
		else
		{
			int rand_ind = rand() % state_list.size();
			next_state = state_list[rand_ind];
		}
	}
	else
	{
		next_state = next_best;
	}
	bool is_best = (next_state - next_best).isZero();
	return make_tuple(next_state, is_best);
}

tuple<ArrayXXf, ArrayXXf, ArrayXXf, ArrayXXf, int, float> play_game(vector<int> gprop, ArrayXXf theta0x, ArrayXXf theta1x, ArrayXXf theta0o, ArrayXXf theta1o, int learner, float gamefrac, bool test, bool pr)
{	
	int R = gprop[0]; int C = gprop[1];
	float gamma = 1; float eps = .1;
	ArrayXXi state2RC = ArrayXXi::Zero(1,2*R*C);
	//cout << "game start" << endl;
	int player = 1;
	int TotTurns = 0;
	int TotDeltaV = 0; //not currently used
	bool cont = true;
	
	ArrayXXi new_state;
	tuple<ArrayXXi, bool> b_m;
	bool is_best = false;
	
	tuple<float, ArrayXXf, ArrayXXf> vdv_x_state;
	tuple<float, ArrayXXf, ArrayXXf> vdv_x_nstate;
	ArrayXXf theta0xtemp;
	ArrayXXf theta1xtemp;
	tuple<float, ArrayXXf, ArrayXXf> vdv_o_state;
	tuple<float, ArrayXXf, ArrayXXf> vdv_o_nstate;
	ArrayXXf theta0otemp;
	ArrayXXf theta1otemp;
	
	while (cont)
	{ 		
		if (player == 1) b_m = best_move(gprop, state2RC, theta0x, theta1x, player, player==learner||learner==0, gamefrac);
		if (player == -1) b_m = best_move(gprop, state2RC, theta0o, theta1o, player, player==learner||learner==0, gamefrac);
		new_state = get<0>(b_m);
		is_best = true;//get<1>(b_m);
		
		//if test and TotTurns in [0,1,2,3,4,5,6,7,8]:
        //    Ans = GetAvgXRes(State,50)
        //    TotDeltaV += np.abs(ValueDV(State, Theta0X, Theta1X, 1)[0] - Ans)

		TotTurns += 1;
		
#pragma omp parallel sections 
		{
		{
		if( (learner == 1 || learner == 0) && is_best && TotTurns >= 1 && TotTurns <= R*C)
		{	
			vdv_x_state = value_dv(gprop, state2RC, theta0x, theta1x, 1);
			vdv_x_nstate = value_dv(gprop, new_state, theta0x, theta1x, 1);
			//cout << "old " << get<0>(vdv_x_state) << endl;
			//cout << "next " << get<0>(vdv_x_nstate) << endl;
			theta0xtemp = theta0x + pow(gamma, R*C - TotTurns) * eps * (get<0>(vdv_x_nstate) - get<0>(vdv_x_state)) * get<1>(vdv_x_state);
			theta1xtemp = theta1x + pow(gamma, R*C - TotTurns) * eps * (get<0>(vdv_x_nstate) - get<0>(vdv_x_state)) * get<2>(vdv_x_state);
			theta0x = theta0xtemp;
			theta1x = theta1xtemp;
			//cout << "new " << get<0>(value_dv(gprop, state2RC, theta0x, theta1x, 1)) << endl;
		}
		}
		#pragma omp section
		{
		if( (learner == -1 || learner == 0) && is_best && TotTurns >= 1 && TotTurns <= R*C)
		{	
			vdv_o_state = value_dv(gprop, state2RC, theta0o, theta1o, -1);
			vdv_o_nstate = value_dv(gprop, new_state, theta0o, theta1o, -1);
			theta0otemp = theta0o + pow(gamma, R*C - TotTurns) * eps * (get<0>(vdv_o_nstate) - get<0>(vdv_o_state)) * get<1>(vdv_o_state);
			theta1otemp = theta1o + pow(gamma, R*C - TotTurns) * eps * (get<0>(vdv_o_nstate) - get<0>(vdv_o_state)) * get<2>(vdv_o_state);
			theta0o = theta0otemp;
			theta1o = theta1otemp;
		}
		}
		}
		if (pr) cout << "hi" << get<0>(value_dv(gprop, new_state, theta0x, theta1x, 1)) << " , " << get<0>(value_dv(gprop, new_state, theta0o, theta1o, -1)) << endl;
		
		cont = ! get<0>(is_over(gprop, new_state));
		state2RC = new_state;
		
		if (pr)
		{
			ArrayXXi stateRC = state2RC.block(0,0,1,R*C) - state2RC.block(0,R*C,1,R*C);
			ArrayXXi stateRCrect(R,C);
			stateRCrect = Map<ArrayXXi>(stateRC.data(),R,C);
			cout << stateRCrect << endl;
		}
		player = -1*player;
	}
	//if (get<1>(is_over(gprop, new_state)) == 1 ) cout << new_state << endl; 
	return make_tuple(theta0x, theta1x, theta0o, theta1o, get<1>(is_over(gprop, new_state)), TotDeltaV/TotTurns);
}

tuple<vector<float>, vector<float>, ArrayXXf, ArrayXXf,  ArrayXXf,  ArrayXXf> learn(vector<int> gprop, int game_num)
{	
	int R = gprop[0]; int C = gprop[1];
	const int hidden = 6*R*C;
	ArrayXXf theta0x = .5 * ArrayXXf::Random(2*R*C+1, hidden);
	ArrayXXf theta1x = .5 * ArrayXXf::Random(hidden+1, 1);
	ArrayXXf theta0o = .5 * ArrayXXf::Random(2*R*C+1, hidden);
	ArrayXXf theta1o = .5 * ArrayXXf::Random(hidden+1, 1);

	tuple<ArrayXXf, ArrayXXf, ArrayXXf, ArrayXXf, int, float> pg;
	
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
			pg = play_game(gprop, theta0x, theta1x, theta0o, theta1o, learner, float(i)/game_num, dotest, true);
			theta0x = get<0>(pg);
			theta1x = get<1>(pg);
			theta0o = get<2>(pg);
			theta1o = get<3>(pg);
			int res = get<4>(pg);
			float avdv = get<5>(pg);
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
			pg = play_game(gprop, theta0x, theta1x, theta0o, theta1o, learner, float(i)/game_num, dotest, false);
			theta0x = get<0>(pg);
			theta1x = get<1>(pg);
			theta0o = get<2>(pg);
			theta1o = get<3>(pg);
			int res = get<4>(pg);
			float avdv = get<5>(pg);
			temp_avdv += avdv;
			if (res == 0) num_ties += 1;
			if (res == 1 ) num_p1Wins += 1;
			//cout << res << endl;
		
		}
	}
	return make_tuple(ties_list, p1Wins_list, theta0x, theta1x, theta0o, theta1o);
}

	

int main()
{
	srand(time(0));
	srand(rand());

	int R = 4; int C = 4; int W = 4;// int hidden = 8*R*C;
	vector<int> gprop; gprop.push_back(R); gprop.push_back(C); gprop.push_back(W);
	//ArrayXXi st2RC(1,2*R*C);
	//st2RC << 1,0,0,1,0,1,1,0,0,  0,1,1,0,1,0,0,0,0;

	//cout << get<0>(is_over(gprop, st2RC)) << endl;
	//cout << get<1>(is_over(gprop, st2RC)) << endl;
	tuple<vector<float>, vector<float>, ArrayXXf, ArrayXXf, ArrayXXf, ArrayXXf> Data = learn(gprop, 15000000);
	//ArrayXXi state2RC(1,2*R*C);
	//state2RC << 1,0,1,0,1,0,0,0,0,0,0,0,0,1,0,1,  0,1,0,0,0,0,0,0,0,1,1,0,0,0,1,0;
	//ArrayXXf th0 = .5 * ArrayXXf::Random(2*R*C+1, hidden);
	//ArrayXXf th1 = .5 * ArrayXXf::Random(hidden+1, 1);
	//cout << state2RC << endl;
	//cout <<"yo " << get<0>(value_dv(gprop, state2RC, th0, th1, 1)) << endl;
	//cout << "yoyo " << get<0>(value_dv(gprop, ArrayXXi::Constant(1,2*R*C,0), th0, th1, 1)) << endl;
	ofstream x0file;
	x0file.open("theta0x.txt");
	x0file << get<2>(Data);
	x0file.close();

	ofstream x1file;
	x1file.open("theta1x.txt");
	x1file << get<3>(Data);
	x1file.close();

	ofstream o0file;
	o0file.open("theta0o.txt");
	o0file << get<4>(Data);
	o0file.close();

	ofstream o1file;
	o1file.open("theta1o.txt");
	o1file << get<5>(Data);
	o1file.close();

	return 0;
}