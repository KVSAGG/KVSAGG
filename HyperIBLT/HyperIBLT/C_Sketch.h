#ifndef _CSKETCH_H
#define _CSKETCH_H

#include "BOBHash32.h"
// #include "BOBHash.h"
#include "params.h"
#include <string.h>
#include <algorithm>

class C_Sketch
{
public:
	int w, d;
	BOBHash32* hash[MAX_HASH_NUM * 2];
	int index[MAX_HASH_NUM];
	double* value[MAX_HASH_NUM];
	int seeds[10],seed_fin;
	// int MAX_CNT, MIN_CNT;

public:
	C_Sketch(int _w, int _d, int hash_seed = 1000)
	{
		d = _d, w = _w;

		for (int i = 0; i < d; i++)
		{
			value[i] = new double[w];
			memset(value[i], 0, sizeof(double) * w);
		}

		// MAX_CNT = (1 << (COUNTER_SIZE - 1)) - 1;
		// MIN_CNT = (-(1 << (COUNTER_SIZE - 1)));

		for (int i = 0; i < d * 2; i++)
		{
			hash[i] = new BOBHash32(i + hash_seed);
			seeds[i] = i + hash_seed;
		}
	}

	void Insert(int x, double y)
	{
		int g = 0;
		for (int i = 0; i < d; i++)
		{
			// index[i] = (MurmurHash3_x86_32((char*)&x, sizeof(uint32_t),seeds[i])) % w;
			index[i] = (hash[i]->run((char*)&x, sizeof(uint32_t))) % w;
			// g = (MurmurHash3_x86_32((char*)&x, sizeof(uint32_t),seeds[i+d])) % 2;
			g = (hash[i + d]->run((char*)&x, sizeof(uint32_t))) % 2;

			if (g == 0)
			{
				
				value[i][index[i]]+=y;
			
			}
			else
			{
				
				value[i][index[i]]-=y;
				
			}
		}
	}


	double Query(int x)
	{
		double temp;
		double res[MAX_HASH_NUM];
		int g;
		for (int i = 0; i < d; i++)
		{
			// index[i] = (MurmurHash3_x86_32((char*)&x, sizeof(uint32_t),seeds[i])) % w;
			index[i] = (hash[i]->run((char*)&x, sizeof(uint32_t))) % w;
			temp = value[i][index[i]];
			// g = (MurmurHash3_x86_32((char*)&x, sizeof(uint32_t),seeds[i+d])) % 2;
			g = (hash[i + d]->run((char*)&x, sizeof(uint32_t))) % 2;

			res[i] = (g == 0 ? temp : -temp);
		}

		sort(res, res + d);
		if (d % 2 == 0)
			return ((res[d / 2] + res[d / 2 - 1]) / 2);
		else
			return (res[d / 2]);
	}

	~C_Sketch()
	{
		for (int i = 0; i < d; i++)
			delete[]value[i];
		for (int i = 0; i < d * 2; i++)
			delete hash[i];
	}
};


#endif
