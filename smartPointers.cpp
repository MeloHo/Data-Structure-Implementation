#include <iostream>
#include <memory> // This is for smart pointers in c++
#include <string>
#include <utility>
#include <vector>

/*

Smart pointer is like a wrapper to a pure/raw pointer.
It frees the memory and destruct itself automatically when it goes out of scope.
Common operators like -> * are overloaded for smart pointers. Use them like usual.

Most examples from:
https://docs.microsoft.com/en-us/cpp/cpp/smart-pointers-modern-cpp?view=vs-2019

unique_ptr:
	Unique pointers.
	Should be explicitly declared.
	Cannot be copied. Cannot share its ownership to its raw pointer.

shared_ptr:
	If some algorithm requires copy elements, use shared_ptr instead of unique_ptr.
	Ownership can be shared across other shared_ptr.
	Contains two members: 1. The raw pointer 2. A pointer to the control block, where the reference counter is strored.
	When ref count = 0, the raw pointer is deleted and memory is freed.

weak_ptr:
	Special case for shared_ptr. Does not increase the reference pointer.
	Usually used to check the existence of an pointed-to object.

*/

class Song
{
public:
	std::string name;
	std::string author;
	Song(){}
	Song(const std::string& newName, const std::string& newAuthor):name(newName),author(newAuthor){}
	~Song(){}
}

// For unique_ptr

std::unique_ptr<Song> SongFactory(const std::string& a, const std::string& b)
{
	return std::make_unique<Song>(a, b);
}

void uniquePtrTest()
{
	// std::make_unique is recommended.
	std::unique_ptr<Song> pSong = std::make_unique<Song>("Mayday", "Gentle");
	std::unique_ptr<Song> target = std::move(pSong);

	std::unique_ptr<Song> newSong = SongFactory("A", "B");
}

void vectorOfSongs()
{
	std::vector<std::unique_ptr<Song>> myV;
	myV.push_back(std::make_unique<Song>("Only one", "Kanye"));
	myV.push_back(std::make_unique<Song>("??", "Chris Brown"));
	// Must pass by reference here. Pass by value is not allowed for there is no copy constructor for unique_ptr
	for(const std::unique_ptr<Song>& song:myV){
		std::cout << "Name: " << song->name << " Author: " << song->author << std::endl;

	}
}

// For shared_ptr

void sharePtrDecleration()
{
	// Method 1. Use std::make_shared. Recommended.
	std::shared_ptr<Song> sp1 = std::make_shared<Song>("love", "James");

	// Method 2. Use new.
	std::shared_ptr<Song> sp2(new Song("love", "James"));

	// Method 3. When initialization must be seperated from declaration:
	std::shared_ptr<Song> sp3(nullptr);
	sp3 = std::make_shared<Song>("Love", "James");
}

void sharePtrInitializaiton()
{
	std::shared_ptr<Song> sp1 = std::make_shared<Song>("love", "James");
	auto sp2(sp1);// Copy constructor;
	auto sp3 = sp1; // Copy operator;
}

void compareTwoSharedPtr()
{
	std::shared_ptr<Song> sp1(new Song("love", "James"));
	std::shared_ptr<Song> sp2(new Song("love", "James"));

	// sp1 == sp2 will give you false even they have same content.
	// This is because sp1 and sp2 are unrelated.

	std::shared_ptr<Song> sp3(sp2);

	// sp2 == sp3 will give you true because they are related.
}








