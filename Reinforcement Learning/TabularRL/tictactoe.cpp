//
//Note: the information below may be invisible in Dev C++. Please select the very row you want to see.

//注：本代码实现了X和O以不同的epsilon值进行左右互搏，两者的胜率均有所提升，
//而由于X方具有巨大的先手优势，因而X的胜率总是远远超过O，但可以看出平局的概率在下降，即二者胜率之和在逼近100%
//要实现X对PolicyDefault或者PolicyRandom的最优策略，只需修改动作策略分支即可
#include <ctime>
#include <iostream>
#include <vector>
#include <map>
#include "tictactoe.hpp"
#include <random>
using namespace std;
// Abstract class
class TicTacToePolicyBase{
    public:
        virtual TicTacToe::Action operator()(const TicTacToe::State& state) = 0;
};
TicTacToe env(false);

// randomly select a valid action for the step.
class TicTacToePolicyRandom : public TicTacToePolicyBase{
    public:
    	//Overload the operator () so that for this bot the expression Policy(State) produces an action
        TicTacToe::Action operator()(const TicTacToe::State& state) {
            vector<TicTacToe::Action> actions = state.action_space();
            int n_action = actions.size();
            int action_id = rand()%n_action;
            return actions[action_id];
        }
        TicTacToePolicyRandom(){
        	srand(unsigned(time(NULL)));
		}
};
class TicTacToePolicyDefault : public TicTacToePolicyBase{
    public:
        TicTacToe::Action operator()(const TicTacToe::State& state){
            vector<TicTacToe::Action> actions = state.action_space();
            if (state.turn == TicTacToe::PLAYER_X){
                // TODO
                
                return actions[0];
            } else {
                return actions[0];
            }
        }
        TicTacToePolicyDefault(){}
};
const double alpha = 0.3;
typedef map<int,double> MP;
// select the first valid action.
class TicTacToePolicyRL : public TicTacToePolicyBase{
    public:
    	MP mp;
        TicTacToe::Action operator()(const TicTacToe::State& state) {
            vector<TicTacToe::Action> actions = state.action_space();
            double max_statevalue = -1.0;//pick the largest value
            vector<TicTacToe::Action>::iterator best_ptr;//pick the most valuable action
        	int current_board = env.get_state().board;
        	bool verbs = env.verbose;
			env.verbose=false;
            for(auto apt=actions.begin();apt!=actions.end();++apt){
            	env.step(*apt);
            	current_board = env.get_state().board;

            	if (mp.find(current_board)==mp.end()){
            		mp[current_board] = 0.5;
            	}
            	if (mp[current_board]>max_statevalue){
            		max_statevalue = mp[current_board];
            		best_ptr = apt;
            	}
				env.step_back();
            }
            env.verbose=verbs;
            current_board = env.get_state().board;
            if (mp.find(current_board)==mp.end()){
            	mp[current_board]=0.5;
            }
            double current_value = mp[current_board];
            mp[current_board] = alpha*max_statevalue+(1-alpha)*current_value;
            return *best_ptr;
        }
        void set_pair(int boards,double value){
        	mp[boards] = value;
		}
};

template <class T>
void Print(T start,T last){
	for(;start!=last;++start){
		cout << *start << endl; 
	}
}
ostream & operator<< (ostream& os,const pair<int,double> &p){
	os << p.first << "," << p.second << " ";
	return os;
}
#include <chrono>
#include <thread>

// randomly select action
const double epsilon_X = 0.05;//Policy X takes e-greedy form
const double epsilon_O = 0.9;//Policy O only takes the Random form
int main(){
	double X_ratio = 0.0;
	double O_ratio = 0.0;
	int X_win_rounds = 0;
	int O_win_rounds = 0;
	TicTacToePolicyRandom policy_Rd;
	TicTacToePolicyDefault policy_Df;
	TicTacToePolicyRL policy_RLX;
	TicTacToePolicyRL policy_RLO;
	srand(unsigned(time(NULL)));
	bool done;
	for(int round=0;round<1000;++round){
		done = false;
		// set verbose true
		
		while (not done){
			TicTacToe::State state = env.get_state();
			TicTacToe::Action action;
			if (state.turn==TicTacToe::PLAYER_X){
				if (double(rand())/RAND_MAX<epsilon_X){
					action = policy_Rd(state);
				}
				else{
					action = policy_RLX(state);
				}
			}
			else if (state.turn==TicTacToe::PLAYER_O){
				if (double(rand())/RAND_MAX<epsilon_O){
					action = policy_Df(state);
				}
				else{
					action = policy_RLO(state);
				}
			}
			env.step(action);
			done = env.done();
			// Deal with the XO action
			int current_stateboard = env.get_state().board;
			if (done){
				if (env.winner()==TicTacToe::PLAYER_X){
					policy_RLX.set_pair(current_stateboard,1.0);
					policy_RLO.set_pair(current_stateboard,0.0);
				}
				else if (env.winner()==TicTacToe::PLAYER_O){
					policy_RLX.set_pair(current_stateboard,0.0);
					policy_RLO.set_pair(current_stateboard,1.0);
				}
				else{
					policy_RLX.set_pair(current_stateboard,0.5);
					policy_RLO.set_pair(current_stateboard,0.5);
				}
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(0));
		}
		X_win_rounds +=  (env.winner()==TicTacToe::PLAYER_X);
		O_win_rounds +=  (env.winner()==TicTacToe::PLAYER_O);
		X_ratio = double(X_win_rounds)/double(round+1);
		O_ratio = double(O_win_rounds)/double(round+1);
		if ((round+1)%100==0){
			cout << "round:" << round+1 << " X_winning_ratio=" << X_ratio << " O_winning_ratio=" << O_ratio << endl;
		}
		env.reset();
	}
	//Print(policy_RL.mp.begin(),policy_RL.mp.end());
	env.verbose=true;
	done=false;
	while (not done){
		TicTacToe::State state = env.get_state();
		TicTacToe::Action action;
		if (state.turn==TicTacToe::PLAYER_X){
			action = policy_RLX(state);
		}
		else if (state.turn==TicTacToe::PLAYER_O){
			action = policy_Df(state);
		}
		env.step(action);
		done = env.done();
		env.print();
	}
    return 0;
};
