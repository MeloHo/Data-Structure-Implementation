/*
unordered_multimap is included in the header <unordered_map>
std::pair, std::make_pair is included in the header <utility>

Cannot use [] in unordered_multimap nor multimap;
map.find(key) will return a random (key, value) pair.

Must break after erase some (key, value) pair

map.equal_range(key) returns a pair of iterator of the range for (key, value1), (key, value2). 

For std::multimap. The element are ordered. Firstly sorted by there key value from small to large.
Then for the pairs with the same key, they are ordered given by the input order.

INT_MAX, INT_MIN is defined in <limits.h>
*/

#include <iostream>
#include <unordered_map>
#include <utility>

typedef std::unordered_multimap<int, int>::iterator ummIt;

int main()
{
  std::unordered_multimap<int, int> myMap;
  myMap.insert(std::make_pair(1,2));
  myMap.insert(std::pair(1,3));
  myMap.insert(std::pair(2,4));
  std::pair<int, int> target(1,2);
  //std::pair<ummIt, ummIt> itPair = myMap.equal_range(1);
  for(auto it = myMap.equal_range(1).first; it != myMap.equal_range(1).second; it++){
    std::pair<int, int> thisOne = *it;
    if(thisOne == target){
      std::cout << "Find! " << it->first << it->second << std::endl; 
      myMap.erase(it);
      break; // Must break here
    }
  }
  std::cout << myMap.count(1) << std::endl;
  
  return 0;
}