#ifndef _FERMAT_H_
#define _FERMAT_H_

#include <iostream>
#include <cstdint>
#include <unordered_map>
#include <queue>
#include <cstring>
#include "BOBHash32.h"
#include "mod.h"
#include "prime.h"
#include <fstream>
using namespace std;

#define DEBUG_F 0

const int cc = 400000;
const int dd = 6568640;
const int rr = 3;


// use a 16-bit prime, so 2 * a mod PRIME will not overflow
// static const uint32_t PRIME_ID = MAXPRIME[24];
// static const uint32_t PRIME_FING = MAXPRIME[24];
static const int PRIME_ID = MAXPRIME[22];
static const int PRIME_FING = MAXPRIME[22];
typedef double vt;

class Fermat {  
public:
    // arrays
    int array_num;
    int entry_num;
    int **id;
    int **fingerprint;
    int **counter;
    vt **value;
    // hash
    BOBHash32 *hash;
    BOBHash32 *hash_fp;
    uint32_t *hash_fing;
    uint32_t (*buckets)[dd];
    uint32_t *table;

    bool use_fing;

// public:
    int pure_cnt;

    // void create_look_up_table() {
    //     table = new uint32_t[PRIME_ID];
    //     for (uint32_t i = 0; i < PRIME_ID; ++i)
    //         table[i] = powMod32(i, PRIME_ID - 2, PRIME_ID);
    // }

    // void clear_look_up_table() {
    //     // delete [] table;
    //     // delete table;
    // }

    void create_array() {
        pure_cnt = 0;
        // id
        id = new int*[array_num];
        for (int i = 0; i < array_num; ++i) {
            id[i] = new int[entry_num];
            memset(id[i], 0, entry_num * sizeof(int));
        }
        // fingerprint
        if (use_fing) {
            fingerprint = new int*[array_num];
            for (int i = 0; i < array_num; ++i) {
                fingerprint[i] = new int[entry_num];
                memset(fingerprint[i], 0, entry_num * sizeof(int));
            }    
        }
        
        // counter
        counter = new int*[array_num];
        for (int i = 0; i < array_num; ++i) {
            counter[i] = new int[entry_num];
            memset(counter[i], 0, entry_num * sizeof(int));
        }

        // value
        value = new vt*[array_num];
        for (int i = 0; i < array_num; ++i) {
            value[i] = new vt[entry_num];
            memset(value[i], 0, entry_num * sizeof(vt));
        }
    }

    void clear_array() {
        for (int i = 0; i < array_num; ++i)
            delete [] id[i];
        delete [] id;

        if (use_fing) {
            for (int i = 0; i < array_num; ++i)
                delete [] fingerprint[i];
            delete [] fingerprint;    
        }
        
        for (int i = 0; i < array_num; ++i)
            delete [] counter[i];
        delete [] counter;

        for (int i = 0; i < array_num; ++i)
            delete [] value[i];
        delete [] value;
    }
    
    Fermat(){
        use_fing=1;
        if (use_fing)
            hash_fp = new BOBHash32(357);
        hash = new BOBHash32[rr];
        for (int i = 0; i < rr; ++i) 
            hash[i].initialize(357 + i + 1);
    }
    Fermat(int r, int c,bool _fing, int _init) : use_fing(_fing) {
        array_num = r;
        entry_num = c;
        create_array();
        
        if (use_fing)
            hash_fp = new BOBHash32(_init+101);
        hash = new BOBHash32[array_num];
        // for (int i = 0; i < array_num; ++i) 
    
        
        hash[0].initialize(_init);
        hash[1].initialize(_init+21);
        hash[2].initialize(_init+51);
    }

    ~Fermat() {
        clear_array();
        if (use_fing)
            delete hash_fp;
        delete [] hash;
    }

    void Insert(int flow_id, int cnt, vt v) {
        if (use_fing) {
            int fing = hash_fp->run((char*)&flow_id, sizeof(int));
            for (int i = 0; i < array_num; ++i) {
                int pos = hash[i].run((char*)&flow_id, sizeof(int)) % entry_num;
                id[i][pos] = (id[i][pos] + mulMod(flow_id, cnt, PRIME_ID)) % PRIME_ID;
                fingerprint[i][pos] = ((uint64_t)fingerprint[i][pos] + mulMod32(fing, cnt, PRIME_FING)) % PRIME_FING;
                counter[i][pos] += cnt;
                value[i][pos] += v;
            }    
        }
        else {
            for (int i = 0; i < array_num; ++i) {
                int pos = hash[i].run((char*)&flow_id, sizeof(int)) % entry_num;
                id[i][pos] = (id[i][pos] + (flow_id * cnt) % PRIME_ID) % PRIME_ID;
                counter[i][pos] += cnt;
                value[i][pos] += v;
            } 
        }
        
    }
    
    void Insert_one(int flow_id, vt v) {
        
        if (use_fing) {
            int fing = hash_fp->run((char*)&flow_id, sizeof(int)) % PRIME_FING;
            for (int i = 0; i < array_num; ++i) {
                int pos = hash[i].run((char*)&flow_id, sizeof(int)) % entry_num;
                id[i][pos] = (id[i][pos] + (flow_id)) ;
                fingerprint[i][pos] = ((int)fingerprint[i][pos] + (int)fing);
                counter[i][pos]++;
                value[i][pos] += v;
            }
        }
        else {
            for (int i = 0; i < array_num; ++i) {
                int pos = hash[i].run((char*)&flow_id, sizeof(int)) % entry_num;
                id[i][pos] = (id[i][pos] + flow_id);
                counter[i][pos]++;
                value[i][pos] += v;
            }
        }
    }

    void Delete_in_one_bucket(int row, int col, int pure_row, int pure_col) {
        
        id[row][col] =  id[row][col] - id[pure_row][pure_col];
        if (use_fing)
            fingerprint[row][col] =  fingerprint[row][col] - fingerprint[pure_row][pure_col];
            
        counter[row][col] -= counter[pure_row][pure_col];
        value[row][col] -= value[pure_row][pure_col];
    }

    
    bool verify(int row, int col, int &flow_id, int &fing) {
        #if DEBUG_F
        ++pure_cnt;
        #endif
        bool flag1=1,flag2=1;
        
        int cnt = counter[row][col];
        if(id[row][col]%cnt) return false;
        if(use_fing){
            if(fingerprint[row][col]%cnt) return false;       
            fing = fingerprint[row][col]/cnt;
        }
        flow_id = id[row][col]/cnt;


        if (!(hash[row].run((char*)&flow_id, sizeof(uint32_t)) % entry_num == col))
            flag1 = false;


        if (use_fing && !(hash_fp->run((char*)&flow_id, sizeof(uint32_t)) % PRIME_FING == fing))
            flag2 = false;

        return min(flag1,flag2);
    }

    void display() {
        cout << " --- display --- " << endl;
        for (int i = 0; i < array_num; ++i) {
            for (int j = 0; j < entry_num; ++j) {
                if (counter[i][j]) {
                    cout << i << "," << j << ":" << counter[i][j] << endl;
                }
            }
        }
    }

    int Decode(unordered_map<int, vt> &result_v) {
        unordered_map<int, int> cont;

        queue<int> *candidate = new queue<int> [array_num];
        int flow_id;
        int fing;
        
        vector<vector<bool>> processed(array_num);
        for (int i = 0; i < array_num; ++i) {
            processed[i].resize(entry_num);
            for (int j = 0; j < entry_num; ++j)
                processed[i][j] = false;
        }
        

        // first round
        for (int i = 0; i < array_num; ++i)
            for (int j = 0; j < entry_num; ++j) {
                if (counter[i][j] == 0) 
                    processed[i][j] = true;
                else if (verify(i, j, flow_id, fing)) { 
                    // find pure bucket
                    processed[i][j] = true;
                    if(result_v.find(flow_id) == result_v.end()){
                        result_v[flow_id] = value[i][j];
                        cont[flow_id] = counter[i][j];
                    }else{
                        result_v[flow_id] += value[i][j];
                        cont[flow_id] += counter[i][j];
                    }
                    for (int t = 0; t < array_num; ++t) {
                        if (t == i) continue;
                        
                        uint32_t pos = hash[t].run((char*)&flow_id, sizeof(uint32_t)) % entry_num;
                        Delete_in_one_bucket(t, pos, i, j);
                        if (t < i)
                            candidate[t].push(pos);
                        processed[t][pos]=false;
                    }
                    Delete_in_one_bucket(i, j, i, j);
                }
            }

        bool pause;
        int timeout=(int)((double)array_num*(double)entry_num*1.5), time=0;
        while (time<timeout) {
            time++;
            pause = true;
            for (int i = 0; i < array_num; ++i) {
                if (!candidate[i].empty()) pause = false;
                while (!candidate[i].empty()) {
                    int check = candidate[i].front();
                    candidate[i].pop();
                    if (processed[i][check]) continue;                           
                    if (counter[i][check] == 0) 
                        processed[i][check] = true;
                
                    else if (verify(i, check, flow_id, fing)) { 
                        // find pure bucket
                        processed[i][check] = true;

                        if(result_v.find(flow_id) == result_v.end()){
                            result_v[flow_id] = value[i][check];
                            cont[flow_id] = counter[i][check];
                        }else{
                            result_v[flow_id] += value[i][check];
                            cont[flow_id] += counter[i][check];
                        }
                        
                        for (int t = 0; t < array_num; ++t) {
                            if (t == i) continue;
                            
                            uint32_t pos = hash[t].run((char*)&flow_id, sizeof(uint32_t)) % entry_num;
                            Delete_in_one_bucket(t, pos, i, check);
                            
                            candidate[t].push(pos);
                            processed[t][pos]=false;

                        }
                        Delete_in_one_bucket(i, check, i, check);
                        
                    }
                }
            }
            if (pause)
                break;
        }
        
        delete [] candidate;
        int fail_cnt=0;
        for (int i = 0; i < array_num; ++i)
            for (int j = 0; j < entry_num; ++j)
                if (!processed[i][j]) 
                    fail_cnt++;        
                    

        return fail_cnt;
    }
};

#endif