#include<iostream>
#include<omp.h>
#include<vector>
#include<random>

using namespace std;


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


double ParallelSum(const vector<double>& arr) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 1; i < arr.size(); ++i) {
        sum += arr[i];
    }
    return sum;
}


double ParallelAverage(const vector<double>& arr) {
    double average = ParallelSum(arr)/arr.size();
    return average;
}


int main() {
    vector<double> arr(10000);

    default_random_engine generator;
    uniform_real_distribution<double> distribution(0.0, 10000.0);

    for (int i = 0; i < 10000; i++) {
    arr[i] = distribution(generator);
    }

    // Print the Array
    // for (auto num : arr){
    //   cout << num << " ";
    // }
    // cout << endl;

    cout << "\nMin     : " << ParallelMin(arr);
    cout << "\nMax     : " << ParallelMax(arr);
    cout << "\nSum     : " << ParallelSum(arr);
    cout << "\nAverage : " << ParallelAverage(arr);
}
