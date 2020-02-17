// Implement Quick sort on an array

#include <iostream>
#include <stdlib.h>

using namespace std;

void swap(int* a, int* b)
{
	int t = *a;
	*a = *b;
	*b = t;
}


/* This function takes last element as pivot, places  
the pivot element at its correct position in sorted  
array, and places all smaller (smaller than pivot)  
to left of pivot and all greater elements to right  
of pivot */
int partition(int arr[], int low, int high)
{
	int pivot = arr[high];
	int i = low-1;
	for(int j=low; j<=high-1; j++)
	{
		if(arr[j]<pivot)
		{
			i++;
			swap(&arr[i], &arr[j]);
		}
	}

	swap(arr[i+1],arr[high]);
	return i+1;
}


/*
This is the main function to do quick sort
Rucursive.
Every time it will put pivot at its final position.
*/
void QuickSort(int arr[], int low, int high)
{
	if(low < high)
	{
		int pivot = partition(arr, low, high);
		QuickSort(arr, low, pivot -1);
		QuickSort(arr, pivot+1, high);
	}
}

/*
This is print function
*/
void printArr(int arr[], int size)
{
	for(int i = 0 ; i<size; i++)
		cout<<arr[i]<<'\t';

	cout<<endl;
}

int main()
{
	int arr[] = {10, 7, 8, 9, 1, 5};
	int n = sizeof(arr) / sizeof(arr[0]);
	printArr(arr , n);
	QuickSort(arr, 0, n-1);
	printArr(arr, n);

	return 0;
}