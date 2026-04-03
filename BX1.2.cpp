#include <iostream>
#include <time.h>
using namespace std;

const int N1 = 2048;

const int REPEAT = 1000; // 循环1000次，拉长实验时间缓解误差

double a1[N1];

// 设置一个全局变量接收结果
double global_sum = 0.0;

int main() {
    clock_t start, finish;

   //初始化
    for (int i = 0; i < N1; i++) {
        a1[i] = 1.0;
    }

    // 1. 平凡算法：单一链式累加
    start = clock();
    for (int r = 0; r < REPEAT; r++) {
        double sum = 0.0;
        for (int i = 0; i < N1; i++) {
            sum += a1[i];
        }
        global_sum += sum; 
    }
    finish = clock();
    cout << "平凡算法耗时: " << (double)(finish - start) / CLOCKS_PER_SEC * 1000 ;

    // 2. 优化算法：两路循环展开
    start = clock();
    for (int r = 0; r < REPEAT; r++) {
        double sum1 = 0.0;
        double sum2 = 0.0;
        for (int i = 0; i < N1; i += 2) {
            sum1 += a1[i];
            sum2 += a1[i + 1];
        }
        double total = sum1 + sum2;
        global_sum += total;
    }
    finish = clock();
    cout << "优化算法耗时: " << (double)(finish - start) / CLOCKS_PER_SEC * 1000;

    return 0;
}