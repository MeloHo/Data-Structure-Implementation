#include<iostream>

template<typename T>
class shared_ptr
{
private:
  T* ptr;
  long* count;
  
  void increment_count()
  {
    ++(*count);
  }
  
  void decrement_count()
  {
    --(*count);
  }
  
public:
  shared_ptr(): ptr(nullptr), count(nullptr){}
  
  explicit shared_ptr(T* incoming): ptr(incoming) {
    count = new long(1);
  }
  
  shared_ptr(const shared_ptr& incoming): ptr(incoming.ptr), count(incoming.count) {
    increment_count();
  }
  
  shared_ptr& operator=(const shared_ptr& incoming) {
    if (ptr != nullptr) {
      decrement_count();
      if ((*count) == 0) {
        delete ptr;
        delete count;
      }
    }
    
    ptr = incoming.ptr;
    count = incoming.count;
    increment_count();
  }
  
  long ref_count() const {
    return *count;
  }
  
  ~shared_ptr()
  {
    if (ptr)
    {
      decrement_count();
      if ((*count) == 0) {
        delete ptr;
        delete count;
      }
    }
    
  }
  
  T* operator->()
  {
    return ptr;
  }
  
  T& operator*()
  {
    return *ptr;
  }
  
  void reset()
  {
    decrement_count();
    if ((*count) == 0) {
      delete ptr;
      delete count;
    }
    ptr = nullptr;
    count = nullptr;
  }
};

class A
{
public:
  int a;
  A(int a_): a(a_){}
  void printA()
  {
    std::cout << "value:" << a << std::endl;
  }
};

int main()
{
  shared_ptr<A> ptr_1(new A(1));
  std::cout << "ref_count:" << ptr_1.ref_count() << std::endl;
  
  shared_ptr<A> ptr_2(ptr_1);
  std::cout << "ref_count:" << ptr_2.ref_count() << std::endl;
  std::cout << "ref_count:" << ptr_1.ref_count() << std::endl;
  
  
  {
    shared_ptr<A> ptr_3 = ptr_1;
    std::cout << "ref_count:" << ptr_3.ref_count() << std::endl;
  }
  
  std::cout << "ref_count:" << ptr_1.ref_count() << std::endl;  
  
  (*ptr_1).printA();
  ptr_1.reset();
  
  std::cout << "ref_count:" << ptr_2.ref_count() << std::endl;
  
  
  return 0;
}