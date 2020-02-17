#include <iostream>
#include <boost/functional/hash.hpp>
#include <unordered_set>
#include <utility>

using namespace std;

typedef pair<int, int> intPair;

int main()
{
  unordered_set<intPair, boost::hash<intPair>> mySet;
  intPair newPair = make_pair(1,2);
  mySet.insert(newPair);
  int count1 = mySet.count(make_pair(1,2));
  int count2 = mySet.count(make_pair(0,2));

  cout << "count1: " << count1 << endl;
  cout << "count2: " << count2 << endl;

  return 0;
}

/*
// This is for unordered_map
#include <iostream>
#include <unordered_map>
#include <utility>
#include <boost/functional/hash.hpp>

using namespace std;

typedef pair<int, int> intPair;

int main()
{
  unordered_map<intPair, int, boost::hash<intPair>> myMap;
  intPair newPair = make_pair(1, 2);
  myMap[newPair] = 1;
  
  cout << myMap.count(newPair) << endl;
  cout << myMap[newPair] << endl;
  
  return 0;
}

*/

