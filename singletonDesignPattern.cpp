#include <iostream>

// This Singleton implementation is lazy initialization:
class Singleton
{
private:
	Singleton(){}
public:
	Singleton(const Singleton&) = delete;
	Singleton& operator=(const Singleton&) = delete;

	static Singleton* getInstance()
	{
		static Singleton* instance = new Singleton();
		return instance;
	}
};

/*
// This is a classic implementation of singleton.

class Singleton
{
private:
	Singleton(){}
	static Singleton* instance;

public:
	static Singleton* getInstance();
	Singleton(const Singleton&) = delete;
	Singleton& operator=(const Singleton&) = delete;
};

Singleton* Singleton::instance = NULL;

Singleton* Singleton::getInstance()
{
	if(instance == NULL){
		instance = new Singleton();
	}
	return instance;
}
*/

int main()
{
	Singleton* s = Singleton::getInstance();
	std::cout << s << std::endl;
}