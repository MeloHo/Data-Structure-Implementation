#include <iostream>
#include <thread>
#include <future>
#include <vector>
#include <functional>

const int MIN_LENGTH = 3;

template<typename Iterator, typename Func>
void parallel_for_each(Iterator first, Iterator last, Func f)
{
  if (first == last) 
  {
    return;
  }
  
  const int length = std::distance(first, last);
  if (length < MIN_LENGTH)
  {
    std::for_each(first, last, f);
    return;
  }
  
  Iterator mid = first;
  std::advance(mid, length / 2);

  std::future<void> leftFuture = std::async(
    &parallel_for_each<Iterator, Func>, first, mid, f
  );
  std::future<void> rightFuture = std::async(
    &parallel_for_each<Iterator, Func>, mid, last, f 
  );
  
  leftFuture.get();
  rightFuture.get();
}

void func(int& a)
{
  a *= 2;
}

// To execute C++, please define "int main()"
int main() {
  std::vector<int> test{1,2,3,4,5};
  
  parallel_for_each<std::vector<int>::iterator, std::function<void(int&)>>(test.begin(), test.end(), func);
  
  for (auto i : test)
  {
    std::cout << i << ",";
  }
  return 0;
}