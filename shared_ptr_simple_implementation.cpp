#include <iostream>

template<class T>
class shared_ptr
{
private:
  T* ptr;
  int* count;
  
  inline void dec_count()
  {
    --(*count);
  }
  
  inline void inc_count()
  {
    ++(*count);
  }
  
public:
  shared_ptr(): ptr(nullptr), count(nullptr) {}
  shared_ptr(T* incoming): ptr(incoming)
  {
    count = new int(1);
  }
  
  ~shared_ptr()
  {
    if (count != nullptr)
    {
      dec_count();
      if (*count == 0)
      {
        delete ptr;
        delete count;
      }
    }
  }
  
  shared_ptr(const shared_ptr& incoming): ptr(incoming.ptr), count(incoming.count)
  {
    inc_count();
  }
  
  shared_ptr& operator=(const shared_ptr& rhs)
  {
    if (count != nullptr)
    {
      dec_count();
      if (*count == 0)
      {
        delete ptr;
        delete count;
      }
    }
    
    ptr = rhs.ptr;
    count = rhs.count;
    inc_count();
    
    return *this;
  }
  
  T& operator->()
  {
    return *ptr;
  }
  
  T* operator*()
  {
    return ptr;
  }
  
  int ref_count() const
  {
    return *count;
  }
  
  void reset()
  {
    if (count != nullptr)
    {
      dec_count();
      if (*count == 0)
      {
        delete ptr;
        delete count;
      }
    }
    
    ptr = nullptr;
    count = nullptr;
  }
};

class A
{
};

int main()
{
  shared_ptr<A> ptr_1(new A());
  std::cout << ptr_1.ref_count() << std::endl;
  
  shared_ptr<A> ptr_2(ptr_1);
  std::cout << ptr_1.ref_count() << std::endl;
  
  {
    shared_ptr<A> ptr_3 = ptr_1;
    std::cout << ptr_3.ref_count() << std::endl;
  }
  
  std::cout << ptr_1.ref_count() << std::endl;
  
  
  return 0;
}