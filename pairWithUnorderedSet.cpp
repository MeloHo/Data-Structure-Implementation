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

/*
// Define own hash

#include <iostream>
#include <boost/functional/hash.hpp>
#include <unordered_map>

template <typename Container>
class CustomHash{
public:
  std::size_t operator()(const Container& con) const{
    return boost::hash_range(con.begin(), con.end());
  }
};

int main() {
  std::vector<std::vector<int>> t{{1, 2}, {3, 4}};
  std::unordered_map<std::vector<int>, int, CustomHash<std::vector<int>>> m;
  for (const auto& t_ : t) {
    m[t_] += 1;
  }
  
  std::cout << m.size() << std::endl;
  std::vector<int> f{1, 2};
  
  if (m.find(f) != m.end()) {
    std::cout << "f in it." << std::endl;
    std::cout << m[f] << std::endl;
  }
  
  return 0;
}*/
