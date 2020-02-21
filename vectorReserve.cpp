/*
This file tests the run time saved by vector::reserve().
If we know the size(max size) beforehand, always use vector::reserve() to save time.

Yidong He
yidongh@andrew.cmu.edu

*/

#include <vector>
#include <chrono>
#include <iostream>

static int SIZE = (int)1e6;

int main(int argc, char const *argv[])
{
	std::vector<int> v1;
	std::vector<int> v2;

	v2.reserve(SIZE);

	auto start1 = std::chrono::high_resolution_clock::now();
	for(int i = 0; i < SIZE; i++){
		v1.push_back(i);
	}
	auto end1 = std::chrono::high_resolution_clock::now();
	auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);

	auto start2 = std::chrono::high_resolution_clock::now();
	for(int i = 0; i < SIZE; i++){
		v2.push_back(i);
	}
	auto end2 = std::chrono::high_resolution_clock::now();
	auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);

	// Compare the time
	std::cout << "Without reserve function: " << duration1.count() << std::endl;
	std::cout << "With reserve function: " << duration2.count() << std::endl;

	return 0;
}