//test1.c
# include <stdio.h>
# include <stdlib.h>
#include <iostream>
#include <cstdint>
#include <unordered_map>
#include <queue>
#include <cstring>
#include "IBLT/fermat1.h"
#include <fstream>
#include <pthread.h>
#include <ctime>
#include <cstdint>
#include <chrono>
#include <unistd.h>
#define MAXTHREADS 32
using namespace std;
using namespace chrono;
struct data{
    int k;
    int idx[3000010];
    vt grad[3000010];
};
struct data_set{
    data din[10];
};
struct IBL_T{
    int r,c,d;
    uint32_t id1[rr][cc];
    uint32_t fingerprint[rr][cc];
    uint32_t counter[rr][cc];
    vt value[rr][cc];
    uint32_t hash_fing[dd];
    uint32_t buckets[rr][dd];
};
extern "C"{
    Fermat** fermat = new Fermat*[100];
    void *deco(void *vargp){

        IBL_T* ibl = ((IBL_T *)vargp);
        
        cout<<"lala1 ";
        Fermat ferm(ibl->r,ibl->c,ibl->id1,ibl->fingerprint,ibl->counter,ibl->value,ibl->hash_fing,ibl->buckets);
        
        cout<<"lala2 ";
        unordered_map<uint32_t, vt> mp;
        cout<<"begin decode ";
        bool suc = ferm.Decode(mp);
        cout<<"lala3 ";

        
        cout<<"end decode "<<suc<<" num:"<<mp.size()<<endl;
        
        return NULL;
    }

    void DDecode_all(IBL_T* iblts,data* res,int num){
        ofstream err("./log.txt",ios::app);
        clock_t begin, endd;
        
        long myid[MAXTHREADS];
        cout<<"START"<<endl;
        auto start = std::chrono::high_resolution_clock::now();
        begin = clock();
        pthread_t tid[MAXTHREADS];
        for(int i=0;i<num;i++){
            myid[i]=i;
            // pthread_create(&tid[i],NULL,deco,&myid[i]);
            pthread_create(&tid[i],NULL,deco,&iblts[i]);
        }
        for(int i=0;i<num;i++){
            pthread_join(tid[i],NULL);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = duration_cast<nanoseconds>(end - start);

        endd = clock();
        cout << "END: " << double(endd - begin) / CLOCKS_PER_SEC * 1000 << "ms" << endl;
        // Fermat ferm(iblt->r,iblt->c,iblt->id1,iblt->fingerprint,iblt->counter,iblt->value,iblt->hash_fing,iblt->buckets);
        err.close();
        return; 
    }

    double CMQuery(int x,Fermat* ferm)
	{
		double temp = 0;
		double min_value = 1e9;
		for (int i = 0; i < 3; i++)
		{
			// index[i] = ferm->buckets[i][x];
			temp = ferm->value[i][ ferm->buckets[i][x] ];
			min_value = temp < min_value ? temp : min_value;
		}
		return min_value;
	}

    void ini_out(int ep){
        string tp1 = "./result/ep"+to_string(ep)+"_IBLT_err.csv";
        string tp2 = "./result/ep"+to_string(ep)+"_CSke_err1.csv";
        string tp3 = "./result/ep"+to_string(ep)+"_CSke_err2.csv";
        ofstream IBLTout(tp1);
        ofstream CSout(tp2);
        ofstream CSoutk(tp3);
        IBLTout<<"cost,MAE,RMSE,"<<endl;
        CSout<<"cost,MAE,RMSE,"<<endl;
        CSoutk<<"cost,MAE,RMSE,"<<endl;
        IBLTout.close();
        CSout.close();
        CSoutk.close();
    }

    void DDecode(IBL_T* iblt,data* res, vt* ori_grad, vt* nz_bincount,int ep){
        
        Fermat ferm(iblt->r,iblt->c,iblt->id1,iblt->fingerprint,iblt->counter,iblt->value,iblt->hash_fing,iblt->buckets);
        unordered_map<uint32_t, vt> mp;

        clock_t begin, endd;
        cout<<"begin decode ";

        auto start = std::chrono::high_resolution_clock::now();

        begin = clock();
        bool suc = ferm.Decode(mp);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = duration_cast<nanoseconds>(end - start);

        endd = clock();
        // cout << "END: " << double(endd - begin) / CLOCKS_PER_SEC * 1000 << "ms" << endl;
        cout<<"end decode "<<suc<<endl;
        unordered_map<uint32_t, vt>::iterator iter;


        double memcost0 = ((double)ferm.entry_num* 12.25* (double)ferm.array_num);
        double memcost = (memcost0/ 1e6);
        int cnt=0;
        
        if(suc){
            double MAE=0,RMSE=0;
            for(iter = mp.begin(); iter != mp.end(); iter++){
                // cout<<iter->first<<" "<<iter->second<<endl;
                res->idx[cnt]=iter->first;
                res->grad[cnt]=iter->second;
                MAE+=fabs(iter->second - ori_grad[iter->first]);
                RMSE+=(iter->second - ori_grad[iter->first])*(iter->second - ori_grad[iter->first]);
                cnt++;
            }
            res->k = cnt;
            // cout<<"IBLT gross absolute error:"<<RMSE/6568640<<endl;
            // cout<<"IBLT gross absolute error:"<<sqrt(RMSE/6568640)<<endl;
            // IBLTout<<memcost<<","<<MAE/6568640<<","<<sqrt(RMSE/6568640)<<","<<endl;
        }else{
            double MAE=0,RMSE=0;
            for(int i=0;i<ep;i++){
                res->idx[i]=i;
                res->grad[i]=CMQuery(i,&ferm);
                MAE+=fabs(CMQuery(i,&ferm) - ori_grad[i]);
                RMSE+=(CMQuery(i,&ferm) - ori_grad[i])*(CMQuery(i,&ferm) - ori_grad[i]);
            }
            cout<<"CM Sketch gross absolute error:"<<RMSE/6568640<<endl;
            // cout<<"CM Sketch gross absolute error:"<<sqrt(RMSE/6568640)<<endl;
            // IBLTout<<memcost<<","<<MAE/6568640<<","<<sqrt(RMSE/6568640)<<","<<endl;
            res->k = ep;
        }


        return; 
    }
    void get_buckets(IBL_T* iblt){
        Fermat fer;
        for(int i=0;i<rr;i++){
            for(int j=0;j<dd;j++){
                iblt->buckets[i][j]= fer.hash[i].run((char*)&j, sizeof(uint32_t)) % cc;
            }  
        }
        for(int j=0;j<dd;j++){
            iblt->hash_fing[j]= fer.hash_fp->run((char*)&j, sizeof(uint32_t)) % 1048573;
        }
    }
    
}
