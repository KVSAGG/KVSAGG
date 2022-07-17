# include <stdio.h>
# include <stdlib.h>
#include <iostream>
#include <cstdint>
#include <unordered_map>
#include <queue>
#include <cstring>
#include "IBLT/fermat3.h"
#include <fstream>
#include <pthread.h>
#include <ctime>
#include <cstdint>
#include <chrono>
#include <unistd.h>
#include "IBLT/C_Sketch.h"
#define MAXTHREADS 32
using namespace std;
using namespace chrono;



int aaa[4]={10000,100000,1000000,10000000};

string bbb[3]={"IBLT","CS","CSK"};
int main(){
        
    for(int j=0;j<3;j++){
        string tp1 = "./speed_data/"+bbb[j]+".csv";
        ofstream fout(tp1);
        fout<<"data,insert,decode,"<<endl;
        for(int k=0;k<4;k++){
            int numIBLT=(int)((double)aaa[k]*1.25/3.0);
            int numCS=(int)((double)numIBLT*12.5/8.0);
            int numCSK=(int)((numIBLT*12.5*3-(double)aaa[k]*4)/8.0/3.0);
                
            double ans_insert=0,ans_decode=0;
            int repeat=10000000/aaa[k];
            if(j==0){
                for(int t=0;t<repeat;t++){
                    Fermat ferm(3,numIBLT,0,t);
                    auto start = std::chrono::high_resolution_clock::now();
                    for(int w=0;w<aaa[k];w++){
                        ferm.Insert_one(w,1);
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    auto insert_duration = duration_cast<nanoseconds>(end - start);                       
                        
                    unordered_map<int, vt> mp;

                    start = std::chrono::high_resolution_clock::now();

                    int fail_cnt= ferm.Decode(mp);

                    end = std::chrono::high_resolution_clock::now();
                    auto decode_duration = duration_cast<nanoseconds>(end - start);                    
                    bool suc = !fail_cnt;         
                      
                    ans_insert+=double(insert_duration.count())/1e6;
                    ans_decode+=double(decode_duration.count())/1e6;

                    
                }
                    
            }
            else if(j==1){
                    
                for(int t=0;t<repeat;t++){
                    C_Sketch cs(numCS, 3,t);
        

                    auto start = std::chrono::high_resolution_clock::now();
                        
                    for(int w=0;w<aaa[k];w++){
                        cs.Insert(w,1);
                    }

                    auto end = std::chrono::high_resolution_clock::now();
                    auto insert_duration = duration_cast<nanoseconds>(end - start);
                        
                    start = std::chrono::high_resolution_clock::now();

                    for(int i=0;i<100*aaa[k];i++){
                        cs.Query(i);
                    }

                    end = std::chrono::high_resolution_clock::now();
                    auto decode_duration = duration_cast<nanoseconds>(end - start);                    
                                 
                    ans_insert+=double(insert_duration.count())/1e6;
                    ans_decode+=double(decode_duration.count())/1e6;

                    
                }
                    
            }
            else if(j==2){
                for(int t=0;t<repeat;t++){
                    C_Sketch cs(numCSK, 3,t);
                    auto start = std::chrono::high_resolution_clock::now();
                        
                    for(int w=0;w<aaa[k];w++){
                        cs.Insert(w,1);
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    auto insert_duration = duration_cast<nanoseconds>(end - start);                       

                    start = std::chrono::high_resolution_clock::now();

                    for(int i=0;i<aaa[k];i++){
                        cs.Query(i);
                    }

                    end = std::chrono::high_resolution_clock::now();
                    auto decode_duration = duration_cast<nanoseconds>(end - start);                    
                               
                    ans_insert+=double(insert_duration.count())/1e6;
                    ans_decode+=double(decode_duration.count())/1e6;

                    
                }
                    
            }

            fout<<aaa[k]<<","<<ans_insert/(double)repeat<<","<< ans_decode/double(repeat)<<","<<endl;
            
            
        }
        fout.close();
    }
    return 0;
}