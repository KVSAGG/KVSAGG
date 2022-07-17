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

const int cc = 300000;
const int dd = 6568640;
const int rr = 3;


// use a 16-bit prime, so 2 * a mod PRIME will not overflow
static const uint32_t PRIME_ID = MAXPRIME[24];
static const uint32_t PRIME_FING = MAXPRIME[24];
typedef double vt;

class Fermat {  
public:
    // arrays
    int array_num;
    int entry_num;
    uint32_t (*id)[cc];
    uint32_t (*fingerprint)[cc];
    uint32_t (*counter)[cc];
    vt (*value)[cc];
    // hash
    BOBHash32 *hash;
    BOBHash32 *hash_fp;
    uint32_t *hash_fing;
    uint32_t (*buckets)[dd];
    uint32_t *table;

    bool use_fing;

// public:
    int pure_cnt;

    
    Fermat(int r, int c,uint32_t _id[rr][cc],uint32_t _fingerprint[rr][cc],uint32_t _counter[rr][cc],double _value[rr][cc],uint32_t _hash_fing[dd],uint32_t _buckets[rr][dd]){
        array_num = r;
        entry_num = c;
        id = _id;
        fingerprint = _fingerprint;
        counter = _counter;
        value = _value;
        hash_fing = _hash_fing;
        buckets = _buckets;
        use_fing=1;
        
    }
    
    Fermat(){
        use_fing=1;
        if (use_fing)
            hash_fp = new BOBHash32(357);
        hash = new BOBHash32[rr];
        for (int i = 0; i < rr; ++i) 
            hash[i].initialize(357 + i + 1);
    }
    

    ~Fermat() {
        
    }

    void Insert(uint32_t flow_id, uint32_t cnt, vt v) {
        if (use_fing) {
            uint32_t fing = hash_fp->run((char*)&flow_id, sizeof(uint32_t));
            for (int i = 0; i < array_num; ++i) {
                uint32_t pos = hash[i].run((char*)&flow_id, sizeof(uint32_t)) % entry_num;
                id[i][pos] = (id[i][pos] + mulMod(flow_id, cnt, PRIME_ID)) % PRIME_ID;
                fingerprint[i][pos] = ((uint64_t)fingerprint[i][pos] + mulMod32(fing, cnt, PRIME_FING)) % PRIME_FING;
                counter[i][pos] += cnt;
                value[i][pos] += v;
            }    
        }
        else {
            for (int i = 0; i < array_num; ++i) {
                uint32_t pos = hash[i].run((char*)&flow_id, sizeof(uint32_t)) % entry_num;
                id[i][pos] = (id[i][pos] + (flow_id * cnt) % PRIME_ID) % PRIME_ID;
                counter[i][pos] += cnt;
                value[i][pos] += v;
            } 
        }
        
    }
    
    void Insert_one(uint32_t flow_id, vt v) {
        
        if (use_fing) {
            uint32_t fing = hash_fp->run((char*)&flow_id, sizeof(uint32_t)) % PRIME_FING;
            for (int i = 0; i < array_num; ++i) {
                uint32_t pos = hash[i].run((char*)&flow_id, sizeof(uint32_t)) % entry_num;
                id[i][pos] = (id[i][pos] + (flow_id % PRIME_ID)) % PRIME_ID;
                fingerprint[i][pos] = ((uint32_t)fingerprint[i][pos] + (uint32_t)fing) % PRIME_FING;
                counter[i][pos]++;
                value[i][pos] += v;
            }
        }
        else {
            for (int i = 0; i < array_num; ++i) {
                uint32_t pos = hash[i].run((char*)&flow_id, sizeof(uint32_t)) % entry_num;
                id[i][pos] = (id[i][pos] + flow_id) % PRIME_ID;
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

    
    bool verify(int row, int col, uint32_t &flow_id, uint32_t &fing) {
        #if DEBUG_F
        ++pure_cnt;
        #endif
        bool flag1=1,flag2=1;
        
        int cnt = counter[row][col];
        
        if(id[row][col]%cnt) return false;
        if(fingerprint[row][col]%cnt) return false;
        flow_id = id[row][col]/cnt;
        fing = fingerprint[row][col]/cnt;
        
        if(!(buckets[row][flow_id]==col))
            flag1 = false;
        if (use_fing && !(hash_fing[flow_id] == fing))
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

    
    bool Decode(unordered_map<uint32_t, vt> &result_v) {
        queue<int> *candidate = new queue<int> [array_num];
        uint32_t flow_id;
        uint32_t fing;
        
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
                    result_v[flow_id] = value[i][j];
                    
                    for (int t = 0; t < array_num; ++t) {
                        if (t == i) continue;
                        uint32_t pos = buckets[t][flow_id];
                        
                        
                        Delete_in_one_bucket(t, pos, i, j);
                        
                        if (t < i)
                            candidate[t].push(pos);
                    }
                    Delete_in_one_bucket(i, j, i, j);
                    
                }
            }
        
        bool pause;
        while (true) {
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
                        result_v[flow_id] = value[i][check];                      
                        
                        for (int t = 0; t < array_num; ++t) {
                            if (t == i) continue;
                            uint32_t pos = buckets[t][flow_id];
                            
                            Delete_in_one_bucket(t, pos, i, check);
                            
                            candidate[t].push(pos);
                        }
                        Delete_in_one_bucket(i, check, i, check);
                        
                    }
                }
            }
            if (pause)
                break;
        }
        delete [] candidate;

        for (int i = 0; i < array_num; ++i)
            for (int j = 0; j < entry_num; ++j)
                if (!processed[i][j]) 
                    return false;        

        return true;
    }
};

#endif