/*
This file implements the priority_queue, especially max-heap using std::vector.
The implementation of max-heap using a linked list is trivial.

Yidong He
yidongh@andrew.cmu.edu
*/

#include <vector>
#include <iostream>
#include <stdexcept>
#include <algorithm>

class PriorityQueue{
private:
  std::vector<int> A;
  int getParent(int i){
    return (i - 1) / 2;
  }
  int getLeftChild(int i){
    return 2 * i + 1;
  }
  int getRightChild(int i){
    return 2 * i + 2;
  }

  void heapDown(int i){
    /*
      This funciton recursively push down a value to the proper position
    */
    if(i >= int(A.size())){
      return;
    }
    int leftChild = getLeftChild(i);
    int rightChild = getRightChild(i);
    int maxIdx = i;
    if(leftChild < int(A.size()) && A[leftChild] > A[i]){
      maxIdx = leftChild;
    }
    if(rightChild < int(A.size()) && A[rightChild] > A[i]){
      maxIdx = rightChild;
    }

    if(maxIdx != i){
      std::swap(A[i], A[maxIdx]);
      heapDown(maxIdx);
    }
  }

  void heapUp(int i){
    /*
      This function recursively push up a value to the proper position
    */
    if(i <= 0){
      return;
    }
    int parent = getParent(i);
    if(A[parent] < A[i]){
      std::swap(A[i], A[parent]);
      heapUp(parent);
    }
  }

public:
  PriorityQueue(){}
  ~PriorityQueue(){}

  PriorityQueue(const PriorityQueue& incoming){
    A.clear();
    for(int i = 0; i < incoming.getSize(); i++){
      A.push_back(incoming.getElement(i));
    }
  }

  PriorityQueue &operator=(const PriorityQueue& incoming){
    A.clear();
    for(int i = 0; i < incoming.getSize(); i++){
      A.push_back(incoming.getElement(i));
    }

    return *this;
  }

  int getSize() const{
    return A.size();
  }

  int getElement(int i) const{
    try{
      if(i >= int(A.size())){
        throw std::out_of_range("Vector<X>::at(): index out of range");
      }
      else{
        return A[i];
      }
    }
    catch(const std::out_of_range& err){
      std::cerr << "Out of Range error: " << err.what() << '\n';
    }
    
    return -1;
  }

  bool empty(){
    return A.empty();
  }

  void push(int val){
    A.push_back(val);
    int index = getSize() - 1;
    heapUp(index);
  }

  void pop(){
    try{
      if(empty()){
        throw std::out_of_range("The heap is empty");
      }
      else{
        A[0] = A.back();
        A.pop_back();
        heapDown(0);
      }
    }
    catch(const std::out_of_range& err){
      std::cerr << "Out of Range error: " << err.what() << '\n';
    }
  }

  int top(){
    try{
      if(empty()){
        throw std::out_of_range("The heap is empty");
      }
      else{
        return A[0];
      }
    }
    catch(const std::out_of_range& err){
      std::cerr << "Out of Range error: " << err.what() << '\n';
    }
    
    return -1;
  }
};

int main(int argc, char const *argv[])
{
  PriorityQueue pq;
  pq.push(1);
  std::cout << pq.top() << std::endl;
  pq.push(2);
  std::cout << pq.top() << std::endl;
  pq.pop();
  std::cout << pq.top() << std::endl;
  pq.push(3);
  std::cout << pq.top() << std::endl;

  return 0;
}


