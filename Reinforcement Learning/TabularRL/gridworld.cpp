#include <utility>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <iomanip>
using namespace std;
class GridWorld{
    public:
        static const int 
            NORTH=0, SOUTH=1, EAST=2, WEST=3;
        static const char ACTION_NAME[][16];
        typedef pair<int, int> State;
        bool verbose;
        State state(){
            return make_pair(x, y);
        }
        void set_state(int x, int y){
            this->x = x;
            this->y = y;
            if (verbose){
                cout << "State reset: (" << x << "," << y << ")" << endl;
            }
        }
        void reset(){
            set_state(0, 0);
        }
        pair<State, double> step(int action){
            State old_state = state();
            double reward = state_transition(action);
            if (verbose){
                cout << "State: (" << old_state.first << "," << old_state.second << ")" << endl;
                cout << "Action: " << ACTION_NAME[action] << endl;
                cout << "Reward: " << reward << endl;
                cout << "New State: (" << x << "," << y << ")" << endl << endl;
            }
            return make_pair(state(), reward);
        }
        int sample_action(){
            return rand() % 4;
        }
        GridWorld(int x=0, int y=0, bool verbose=false){
            set_state(x, y);
            this->verbose = verbose;
        }
        
    private:
        int x, y;
        double state_transition(int action){
            if (state() == make_pair(1, 0)){
                x = 1;
                y = 4;
                return 10.0;
            }
            if (state() == make_pair(3, 0)){
                x = 3;
                y = 2;
                return 5.0;
            }
            if (action == NORTH and y == 0 or
                action == SOUTH and y == 4 or
                action == EAST and x == 4 or
                action == WEST and x == 0){
                return -1.0; 
            }
            switch (action){
                case NORTH:
                    y --; break;
                case SOUTH:
                    y ++; break;
                case EAST:
                    x ++; break;
                case WEST:
                    x --; break;
            }
            return 0.0;
        }
};
const char GridWorld::ACTION_NAME[][16] = {"NORTH(0,-1)", "SOUTH(0,1)", "EAST:(1,0)", "WEST:(-1,0)"};
double values[5][5];
double delta_r,r,max_dr,epsilon=0.001;
#include <chrono>
#include <thread>
//V_pi(s)
void EquiprobablePolicy(){
	while (true){
   		max_dr=0.0;
		for(int x=0;x<=4;++x){
			for(int y=0;y<=4;++y){
				delta_r=0.0;
				r=0.0;
				if(x==1 and y==4){
					r=10+values[1][0];
				} 
				else if(x==3 and y==4){
					r=5+values[3][2];
				}
				else{
					if(x==0) r-=0.25;
					else r+=values[x-1][y]/4;
					if(y==0) r-=0.25;
					else r+=values[x][y-1]/4;
					if(x==4) r-=0.25;
					else r+=values[x+1][y]/4;
					if(y==4) r-=0.25;
					else r+=values[x][y+1]/4;
				}
				
				delta_r = fabs(r-values[x][y]);
				max_dr = max(delta_r,max_dr);
				values[x][y]=r;
			}
		}
		if (max_dr<epsilon){
			break;
		}

	}
	for(int j=0;j<=4;++j){
		for(int i=0;i<=4;++i){
			cout << fixed << setprecision(2);
			cout << values[i][4-j] << " ";
		}
		cout << endl;

   }
}

int policy[5][5];
bool change;
int main(){
    for (int i=0;i<=4;i++){
    	for (int j=0;j<=4;j++){
    		values[i][j]=0.0;
    		policy[i][j]=0;
		}
	}
    cout << endl;
    EquiprobablePolicy();
    for (int i=0;i<=4;i++){
    	for (int j=0;j<=4;j++){
    		values[i][j]=0.0;
    		policy[i][j]=0;
		}
	}
    cout << endl;
	while (true){
		while (true){
	    	max_dr=0.0;
			for(int x=0;x<=4;++x){
				for(int y=0;y<=4;++y){
					delta_r=0.0;
		    		r=0.0;
		    		if(x==1 and y==4){
		    			r=10+values[1][0];
					} 
					else if(x==3 and y==4){
						r=5+values[3][2];
					}
					else{
						int action = policy[x][y];
						switch(action){
							case 0:
                                if (x==0){r=-1;}
                                else{
                                    r = values[x-1][y]*0.9;
                                }
								break;
							case 1:
                                if (y==0){r=-1;}
								else{
                                    r = values[x][y-1]*0.9;
                                }
								break;
							case 2:
                                if (x==4){r=-1;}
								else{
                                    r = values[x+1][y]*0.9;
                                }
								break;
							case 3:
                                if (y==4){r=-1;}
                                else{
                                    r = values[x][y+1]*0.9;
                                }
								break;
						}
					}
					delta_r = fabs(r-values[x][y]);
					max_dr = max(delta_r,max_dr);
					values[x][y]=r;
				}
			}
			if (max_dr<epsilon){
				break;
			}
		}
		bool stable=true;
		int max_act;
		double max_value;
		for(int x=0;x<=4;x++){
			for(int y=0;y<=4;y++){
				max_act=0;
				max_value=0.0;
				if(x!=1 or y!=4 or x!=3){
					if(x>0 and values[x-1][y]>max_value){
					max_act=0;//WEST
					max_value=values[x-1][y];
					}
					if(y>0 and values[x][y-1]>max_value){
						max_act=1;//SOUTH
						max_value=values[x][y-1];
					}
					if(x<4 and values[x+1][y]>max_value){
						max_act=2;//EAST
						max_value=values[x+1][y];
					}
					if(y<4 and values[x][y+1]>max_value){
						max_act=3;//NORTH
						max_value=values[x][y+1];
					}
				}
				if (max_act!=policy[x][y]){
					stable=false;
					policy[x][y]=max_act;
				}
			}
		}
		if(stable){
			break;
		}
	}
	for(int j=0;j<=4;++j){
		for(int i=0;i<=4;++i){
			cout << values[i][4-j] << " ";
		}
		cout << endl;

    }
    return 0;
}