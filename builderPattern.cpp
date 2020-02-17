/*
This file demonstrates the builder design pattern for c++.
*/

#include <iostream>

class foo
{
public:
	class builder;

	foo(int prop1, bool prop2, bool prop3, std::vector<int> prop4)
	:prop1(prop1), prop2(prop2), prop3(prop3), prop4(prop4){}

	int prop1;
	bool prop2;
	bool prop3;
	std::vector<int> prop4;
}

class foo::builder
{
public:
	builder& set_prop1(int val){
		prop1 = val;
		return *this;
	}
	builder& set_prop2(bool val){
		prop2 = val;
		return *this;
	}
	builder& set_prop3(bool val){
		prop3 = val;
		return *this;
	}
	builder& set_prop4(std::vector<int> val){
		prop4 = val;
		return *this;
	}
	foo build() const
	{
		return foo(prop1, prop2, prop3, prop4);
	}
private:
	int prop1 = 0;
	bool prop2 = false;
	bool prop3 = false;
	std::vector<int> prop4 = {};

}

int main()
{
	foo f = foo::builder().set_prop1(5).set_prop3(false).build();
}