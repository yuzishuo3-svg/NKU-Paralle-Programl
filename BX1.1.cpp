#include<iostream>
#include<time.h>
using namespace std;
const int N = 2048;
const int REPEAT = 2000;//循环次数，方便计时
double b[N][N];
double a[N];
double sum[N];
int main() {
	clock_t start, finish;//用于计时
	//初始化
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			b[i][j] = 2.0;
		}
	}for (int i = 0; i < N; i++) {
		a[i] = 1.0;
	}
	//平凡算法
	start = clock();
	for (int i = 0; i < REPEAT; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				sum[j] += b[k][j] * a[k];
			}
		}
	}
	finish = clock();
	cout << "优化前的耗时" << (double)(finish - start) / CLOCKS_PER_SEC * 1000;
	//优化算法
	start = clock();
	for (int i = 0; i < REPEAT; i++) {
		for (int j = 0; j < N; j++) sum[j] = 0.0;
		for (int k = 0; k < N; k++) {
			for (int l = 0; l < N; l++) {
				sum[l] += b[k][l] * a[k];
			}
		}
	}
	finish = clock();
	cout << "优化后的耗时" << (double)(finish - start) / CLOCKS_PER_SEC * 1000;
}