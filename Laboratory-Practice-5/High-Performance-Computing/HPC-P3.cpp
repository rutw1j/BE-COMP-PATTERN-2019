#include<iostream>
#include<vector>
#include<omp.h>

using namespace std;

// Function to find minimum value in an array
double ParallelMin(const vector<double>& arr) {
    double min_value = arr[0];
    #pragma omp parallel for reduction(min:min_value)
    for (size_t i = 1; i < arr.size(); ++i) {
        if (arr[i] < min_value) {
            min_value = arr[i];
        }
    }
    return min_value;
}

// Function to find maximum value in an array
double ParallelMax(const vector<double>& arr) {
    double max_value = arr[0];
    #pragma omp parallel for reduction(max:max_value)
    for (size_t i = 1; i < arr.size(); ++i) {
        if (arr[i] > max_value) {
            max_value = arr[i];
        }
    }
    return max_value;
}

// Function to calculate sum of values in an array
double ParallelSum(const vector<double>& arr) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < arr.size(); ++i) {
        sum += arr[i];
    }
    return sum;
}

// Function to calculate average of values in an array
double ParallelAverage(const vector<double>& arr) {
    double average = ParallelSum(arr);
    return average / arr.size();
}


int main() {
    vector<double> arr = {1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0};

    cout << "\n" << "Input Array" << "\n|  ";
    for (auto num : arr)
        cout << num << "  |  ";

    cout << "\n\nMin      : " << ParallelMin(arr);
    cout << "\nMax      : " << ParallelMax(arr);
    cout << "\nSum      : " << ParallelSum(arr);
    cout << "\nAverage  : " << ParallelAverage(arr);
}