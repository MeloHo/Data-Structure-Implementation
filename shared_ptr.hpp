// practice for shared pointer
// Source: https://github.com/SRombauts/shared_ptr/blob/master/include/shared_ptr.hpp
#ifndef _SHARED_PTR_HPP
#define _SHARED_PTR_HPP

#include <iostream>
#include <utility>
#include <cstddef>
#include <algorithm>
#include <cassert>
#define SHARED_ASSERT(x) aseert(x)

class shared_prt_count
{
public:
	shared_prt_count(): pn(NULL)
	{
	}
	shared_prt_count(const shared_prt_count& count): pn(count.pn)
	{
	}

	void swap(shared_prt_count& lhs) throw() // never throws
	{
		std::swap(pn, lhs.pn);
	}

	long use_count(void) const throw()
	{
		long count = 0;
		if (pn != NULL)
		{
			count = *pn;
		}

		return count;
	}

	template<class U>
	void acquire(U* p)
	{
		if (p != NULL)
		{
			if (pn == NULL)
			{
				try
				{
					pn = new long(1);
				}
				catch(std::bad_alloc&)
				{
					delete p;
					throw;
				}
			}
			else
			{
				++(*pn);
			}
		}
	}

	template<class U>
	void release(U* p) throw() // never throws
	{
		if (pn != NULL)
		{
			--(*pn);
			if (*pn == 0)
			{
				delete p;
				delete pn;
			}
			pn = NULL;
		}
	}

public:
	long* pn;
};

template<class T>
class shared_ptr
{
public:
	typedef T element_type;

	shared_ptr(void) throw() : // never throws
		px(NULL),
		pn()
	{
	}

	explicit shared_ptr(T* p) : // may throw std::bad_alloc
		pn()
	{
		acquire(p);
	}

	shared_ptr(const shared_ptr& ptr) throw() : // never throws
		pn(ptr.pn)
	{
		SHARED_ASSERT((ptr.px == NULL) || (ptr.pn.use_count() != 0));
		acquire(ptr.px);
	}

	shared_ptr& operator=(shared_ptr ptr) throw()
	{
		swap(ptr);
		return *this;
	}

	~shared_ptr(void) throw()
	{
		release();
	}

	void reset(void) throw()
	{
		release();
	}

	void reset(T* p)
	{
		SHARED_ASSERT((p == NULL) || (px != p));
		release();
		acquire(p);
	}

	void swap(shared_ptr& lhs) throw()
	{
		std::swap(px, lhs.px);
		pn.swap(lhs.pn);
	}

	long use_count(void) const throw()
	{
		return pn.use_count();
	}

	T* get(void) const throw()
	{
		return px;
	}

	T& operator*() const throw()
	{
		SHARED_ASSERT(px != NULL);
		return *px;
	}

	T* operator->() const throw()
	{
		SHARED_ASSERT(px != NULL);
		return px;
	}

private:
	void acquire(T* p)
	{
		pn.acquire(p);
		px = p;
	}

	void release(void) throw() // never throws
	{
		pn.release(px);
		px = NULL;
	}



private:
	T* px;
	shared_prt_count pn;
};

#endif