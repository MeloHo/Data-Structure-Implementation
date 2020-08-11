// practice for implementing unique_ptr
#ifndef _UNIQUE_PTR_HPP
#define _UNIQUE_PTR_HPP

#include <iostream>
#include <utility>

template<typename T>
class unique_ptr
{
private:
  T* pointer;

public:
  unique_ptr(): pointer(nullptr){}
  explicit unique_ptr(T* data): pointer(data){}

  unique_ptr(const unique_ptr& input) = delete;
  unique_ptr& operator=(const unique_ptr& rhs) = delete;

  ~unique_ptr()
  {
    if (pointer != nullptr)
    {
      delete pointer;
    }
  }

  unique_ptr(unique_ptr&& input) noexcept : pointer(input.release())
  {
  }

  unique_ptr& operator=(unique_ptr&& input) noexcept
  {
    reset(input.release());
    return *this;
  }

  T* operator->() const {return pointer;}
  T& operator*() const {return *pointer;}

  T* get() const {return pointer;}
  T* release() noexcept
  {
    T* result = nullptr;
    std::swap(pointer, result);
    return result;
  }

  void reset()
  {
    T* tmp = release();
    delete tmp;
  }

  void reset(T* pointer_)
  {
    delete pointer;
    pointer = pointer_;
  }

  explicit operator bool() noexcept
  {
    return pointer != nullptr;
  }
  
  void swap(unique_ptr& input) noexcept
  {
    std::swap(pointer, input.pointer);
  }
};


#endif