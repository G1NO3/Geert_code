#include <chrono>
#include <thread>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <utility>
#include <vector>
#include "blackjack.hpp"
#include<bits/stdc++.h>
class BlackjackPolicyBase{
    public:
        virtual int operator() (const Blackjack::State& state)=0;
};

//class BlackjackPolicyDefault : public BlackjackPolicyBase{
//    public:
//        int operator() (const Blackjack::State& state){
//            if (state.turn == Blackjack::PLAYER){
//                return state.player_sum >= 20 ? Blackjack::STICK : Blackjack::HIT;
//            } else {
//                return state.dealer_sum >= 17 ? Blackjack::STICK : Blackjack::HIT;
//            }
//        }
//};

class BlackjackPolicyLearnableDefault : public BlackjackPolicyBase{
    static constexpr const char *ACTION_NAME = "HS";
    public:
        int operator() (const Blackjack::State& state){
            if (state.turn == Blackjack::DEALER){
                return state.dealer_sum >= 17 ? Blackjack::STICK : Blackjack::HIT;
            } else {
                return policy[state.dealer_shown][state.player_ace][state.player_sum];
            }
        }
        void update_policy(){
            for (int dealer_shown = 1; dealer_shown <= 10; ++ dealer_shown){
                for (int player_ace = 0; player_ace <= 1; ++ player_ace){
                    for (int player_sum = 12; player_sum <= 21; ++ player_sum){
                    	policy[dealer_shown][player_ace][player_sum] = \
                    			value[dealer_shown][player_ace][player_sum][0]>=\
								value[dealer_shown][player_ace][player_sum][1]?0:1;
                    }
                }
            }
            // simply take argmax
        }
        // n iterations for each start (initial state and first action), no need to modify.
        void update_value(Blackjack& env,int n=10000){
        	// dealer_shown,player_ace,player_sum,action
        	vector<EpisodeStep> path_state;
        	Blackjack::StepResult result;
        	EpisodeStep onestep;
        	int newact,length,qsize;
        	double qsum,rewards;
			for (int dealer_shown = 1; dealer_shown <= 10; ++ dealer_shown){
				for (int player_ace = 0; player_ace <= 1; ++ player_ace){
					for (int player_sum = 12; player_sum <= 21; ++ player_sum){
						for (int act=0;act<=1;++ act){
							for (int it=0;it<n;++ it){
								path_state.clear();
								env.reset(dealer_shown,player_ace,player_sum);
								onestep = EpisodeStep(env.state(),act);
								path_state.push_back(onestep);
								result=env.step(act);
								while (not result.done){
									Blackjack::State state_now = result.state;
									if (state_now.turn == Blackjack::PLAYER){
										newact = policy[result.state.dealer_shown][result.state.player_ace][result.state.player_sum];
										onestep = EpisodeStep(result.state,newact);
										path_state.push_back(onestep);
										result=env.step(newact);
									}
									else{
										newact = state_now.dealer_sum >= 17 ? Blackjack::STICK : Blackjack::HIT;
										result = env.step(newact);
									}
								}
								rewards = result.player_reward;
								length = path_state.size();
								for (int l=0;l<length;++l){
									qtable[path_state[l].dealer_shown][path_state[l].player_ace][path_state[l].player_sum][path_state[l].action].push_back(rewards);
								}
							}
						}

					}
				}
            }
			for (int dealer_shown = 1; dealer_shown <= 10; ++ dealer_shown){
				for (int player_ace = 0; player_ace <= 1; ++ player_ace){
					for (int player_sum = 12; player_sum <= 21; ++ player_sum){
						for (int act=0;act<2;++ act){
							qsize = qtable[dealer_shown][player_ace][player_sum][act].size();
							qsum = 0.0;
							for (int l=0;l<qsize;++l){
								qsum += qtable[dealer_shown][player_ace][player_sum][act][l];
							}
							value[dealer_shown][player_ace][player_sum][act] = qsum/qsize;
						}
					}
				}
			}
        }
            // REMEMBER to call set_value_initial() at the beginning
            // simulate from every possible initial state:(dealer_shown, player_ace, player_sum) \
            //    (call Blackjack::reset(,,) to do this) and player's every possible first action
            // BE AWARE only use player's steps (rather than dealer's) to update value estimation. 
        	// Exploring Starting
        
        void print_policy() const {
            cout << setw(10) << "Player Without Ace" << "\t\t" << "Player With Usable Ace." << endl;
            for (int player_sum = 21; player_sum >= 11; -- player_sum){
                for (int dealer_shown = 1; dealer_shown <= 10; ++ dealer_shown){
                    cout << ACTION_NAME[policy[dealer_shown][0][player_sum]];
                }
                cout << "\t\t\t\t" ;
                for (int dealer_shown = 1; dealer_shown <= 10; ++ dealer_shown){
                    cout << ACTION_NAME[policy[dealer_shown][1][player_sum]];
                }
                cout << endl;
            }
            cout << endl;
        }
        void print_value() const {
            cout << setw(40) << "Player Without Ace" << setw(20) <<"\t\t\t" << "Player With Usable Ace." << endl;
            for (int player_sum = 21; player_sum >= 11; -- player_sum){
                for (int dealer_shown = 1; dealer_shown <= 10; ++ dealer_shown){
                    cout << fixed << setprecision(2) << setw(6) << value[dealer_shown][0][player_sum][
                        policy[dealer_shown][0][player_sum]];
                }
                cout << "\t";
                for (int dealer_shown = 1; dealer_shown <= 10 ; ++ dealer_shown){
                    cout << fixed << setprecision(2) << setw(6) << value[dealer_shown][1][player_sum][
                        policy[dealer_shown][1][player_sum]];
                }
                cout << endl;
            }
            cout << endl;
        }
        
        BlackjackPolicyLearnableDefault(){
        	std::default_random_engine e(time(0));
        	uniform_int_distribution<int> u(0,1);
            for (int dealer_shown = 1; dealer_shown <= 10; ++ dealer_shown){
                for (int player_ace = 0; player_ace <= 1; ++ player_ace){
                    for (int player_sum = 0; player_sum <= 21; ++ player_sum){
                        int& action = policy[dealer_shown][player_ace][player_sum];
                        if (player_sum<=11){
                        	action = Blackjack::HIT;
						}
						else{
							action = u(e);//Initialize the policy randomly
						}
                    }
                }
            }
        }
    private:
        // 11: dealer_shown (A-10); 
        // 2: player_usable_ace (true/false);
        // 22: player_sum (only need to consider 11-20, because HIT when sum<11 and STICK when sum=21 are always best action.)
        int policy[11][2][22];
        // 11:dealer_shown; 2:player_usable_ace; 22:player_sum; 2:action (0:HIT, 1:STICK)
        double value[11][2][22][2];
        int state_action_count[11][2][22][2];
		array< array <array <array <vector<double>,2>,22>,2>,11> qtable;

        // record a episode sampled (only player's steps).
        struct EpisodeStep{
            // state: (dealer_shown, player_ace, player_sum)
            int dealer_shown;
            int player_ace;
            int player_sum;
            // the action taken at state
            int action;
            EpisodeStep(const Blackjack::State& state, int action){
                dealer_shown = state.dealer_shown;
                player_ace = int(state.player_ace);
                player_sum = state.player_sum;
                this->action = action;

            }
            void print(){
            	cout << dealer_shown << "" << player_ace << " " << player_sum << " " << action;
            	cout << endl;
			}
            EpisodeStep(){}
        };
        vector<EpisodeStep> episode; 

        void set_value_initial(){
            memset(value, 0, sizeof(value));
            memset(state_action_count, 0, sizeof(state_action_count));
        }
};

// Demonstrative play of player & dealer with default policy 
int main(){
    BlackjackPolicyLearnableDefault policy;
    Blackjack env(false);
    for (int iteration=0;iteration<20;++iteration){
    	policy.update_value(env,1000);
    	policy.update_policy();
    }
    policy.print_policy();
    bool done=false;
    env.verbose=true;
    env.reset();
    int action;
    Blackjack::StepResult result;
    while (not result.done){
    	action = policy(env.state());
    	result = env.step(action);
    }
    return 0;
}
