#include <iostream>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;


void merge(int arr[], int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;
    int L[n1], R[n2];

    for (int i = 0; i < n1; i++) 
        L[i] = arr[l + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[ m + 1 + j];

    int i = 0, j = 0, k = l;
    while(i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}


// Sequential Merge Sort
void MergeSort(int arr[], int l, int r) {
    if (l < r) {
        int m = l + (r-l) / 2;
        MergeSort(arr, l, m);
        MergeSort(arr, m+1, r);
        merge(arr, l, m, r);
    }
}


// Parallel Merge Sort
void ParallelMergeSort(int arr[], int l, int r) {
    if (l < r) {
        int m = l + (r-l) / 2;
        #pragma omp parallel section
        {
            #pragma omp  section
            ParallelMergeSort(arr, l, m);
            #pragma omp section
            ParallelMergeSort(arr, m+1, r);
        }
        merge(arr, l, m, r);
    }
}


int main() {

    const int size = 10000;
    int arr[size], arr_copy[size];

    // Initialize input values
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 10000;
        arr_copy[i] = arr[i];
    }

    auto start = high_resolution_clock::now();
    MergeSort(arr_copy, 0, size-1);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end-start);
    cout << "\n" << "Sequential Merge Sort  : " << duration.count() / 1000.0;

    // Reinitialize input values
    for (int i = 0; i < size; i++) {
        arr_copy[i] = arr[i];
    }

    start = high_resolution_clock::now();
    ParallelMergeSort(arr_copy, 0, size-1);
    end = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end-start);
    cout << "\n" << "Parallel Merge Sort    : " << duration.count() / 1000.0;
}