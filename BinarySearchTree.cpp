#include <iostream>
#include <map>
#include <vector>
using namespace std;

class TreeNode
{
  public:
  int val;
  int height;
  TreeNode* left;
  TreeNode* right;
  
  TreeNode(){}
  TreeNode(int newVal): val(newVal), height(0), left(NULL), right(NULL){}
  
};

class BST
{
  public:
  TreeNode* root;
  
  BST(){root = nullptr;}
  void add(int val);
  void initializeHeight();
  void printReverse();
  void del(int val);
};

void BST::add(int val){
  if(root == NULL){
    root = new TreeNode(val);
    return;
  }
  TreeNode* prev = NULL, *cur = root;
  while(cur != NULL){
    if(cur->val > val){
      prev = cur;
      cur = cur->left;
    }
    else{
      prev = cur;
      cur = cur->right;
    }
  }
  TreeNode* newNode = new TreeNode(val);
  if(val > prev->val) prev->right = newNode;
  else prev->left = newNode;
}

int findLeftMostInRight(TreeNode* root)
{
  TreeNode* cur = root;
  while(cur->left){
    cur = cur->left;
  }
  return cur->val;
}

TreeNode* deleteNode(TreeNode* root, int val)
{
  if(root == NULL) return NULL;
  else if(root->val != val){
    root->left = deleteNode(root->left, val);
    root->right = deleteNode(root->right, val);
    return root;
  }
  else{
    if(!root->left && !root->right) return NULL;
    else if((!root->left && root->right) || (!root->right && root->left)){
      return root->left?root->left:root->right;
    }
    else{
      int newVal = findLeftMostInRight(root->right);
      root->val = newVal;
      root->right = deleteNode(root->right, newVal);
      return root;
    }
  }
}

void BST::del(int val)
{
  root = deleteNode(root, val);
}

int getHeight(TreeNode* root)
{
  if(root == NULL) return 0;
  int leftHeight = getHeight(root->left), rightHeight = getHeight(root->right);
  root->height = max(leftHeight, rightHeight) + 1;
  return root->height;
}

void BST::initializeHeight()
{
  int dummy = getHeight(root);
}

void pushToMap(TreeNode* root, multimap<int, int>& myMap)
{
  if(root == NULL) return;
  myMap.insert(make_pair(root->height, root->val));
  pushToMap(root->left, myMap);
  pushToMap(root->right, myMap);
}

void printFromMap(const multimap<int, int>& myMap)
{
  vector<vector<int>> result;
  multimap<int, int>::const_iterator it = myMap.begin();
  while(it != myMap.end())
  {
    vector<int> temp;
    int curHeight = (*it).first;
    while(it != myMap.end() && (*it).first == curHeight)
    {
      temp.push_back((*it).second);
      it++;
    }
    result.push_back(temp);
  }
  
  for(int i = 0; i < (int)result.size(); i++)
  {
    for(int j = 0; j < (int)result[i].size(); j++)
    {
      cout << result[i][j] << " ";
    }
    cout << endl;
  }
}

void BST::printReverse()
{
  initializeHeight();
  multimap<int, int> myMap;
  pushToMap(root, myMap);
  printFromMap(myMap);
}

int main()
{
  BST binaryTree;
  binaryTree.add(5);
  binaryTree.add(7);
  binaryTree.add(4);
  binaryTree.add(2);
  binaryTree.add(6);
  binaryTree.add(8);
  binaryTree.add(1);
  binaryTree.add(3);
  binaryTree.printReverse();
  
  binaryTree.del(5);
  binaryTree.printReverse();
}