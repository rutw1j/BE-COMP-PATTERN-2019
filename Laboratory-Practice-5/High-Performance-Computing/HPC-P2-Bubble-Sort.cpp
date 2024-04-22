#include<iostream>
#include<omp.h>
#include<chrono>

using namespace std;
using namespace std::chrono;


// Sequential Bubble Sort
void BubbleSort(int arr[], int arr_size) {
    for (int i = 0; i < arr_size-1; i++) {
        for (int j = 0; j < arr_size-i-1; j++) {
            if (arr[j] > arr[j+1]) {
                int temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
        }
    }
}


// Parallel Bubble Sort
void ParallelBubbleSort(int arr[], int arr_size) {
    #pragma omp parallel for
    for (int i = 0; i < arr_size-1; i++) {
        for (int j = 0; j < arr_size-i-1; j++) {
            if (arr[j] > arr[j+1]) {
                int temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
        }
    }
}


int main() {

    const int arr_size = 10000;
    int arr[arr_size], arr_copy[arr_size];

    // Initialize input arr
    srand(time(NULL));
    for (int i = 0; i < arr_size; i++) {
        arr[i] = rand() % 10000;
        arr_copy[i] = arr[i];
    }

    auto start = high_resolution_clock::now();
    BubbleSort(arr_copy, arr_size);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end-start);
    cout << "\n" << "Sequential Bubble Sort  : " << duration.count() / 1000.0 << " seconds";

    // Reinitialize input arr
    for (int i = 0; i < arr_size; i++) {
        arr_copy[i] = arr[i];
    }

    start = high_resolution_clock::now();
    ParallelBubbleSort(arr_copy, arr_size);
    end = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end-start);
    cout << "\n" << "Parallel Bubble Sort    : " << duration.count() / 1000.0 << " seconds";
}