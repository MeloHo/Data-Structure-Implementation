// Realization of Linked List. Add. Find. Delete. Print. Reverse. Sort.
#include<iostream>

using namespace std;

class ListNode{
  public:
  int val;
  ListNode* next;
  ListNode(int newVal):val(newVal), next(NULL){}
};

class LinkedList{
  public:
  ListNode* head;
  void add(int val);
  bool find(int val);
  void del(int val);
  void print();
  void reverse();
  void sortMerge();
  void sortSelection();
  
  LinkedList(){head = NULL;}
};

void LinkedList::add(int val){
  if(head == NULL){
    ListNode* newNode = new ListNode(val);
    head = newNode;
  }
  else{
    ListNode* cur = head;
    while(cur->next){
      cur = cur->next;
    }
    ListNode* newNode = new ListNode(val);
    cur->next = newNode;
  }
}

bool LinkedList::find(int val){
  ListNode* cur = head;
  while(cur != NULL){
    if(cur->val == val){
      return true;
    }
    cur = cur->next;
  }
  return false;
}

ListNode* deleteNode(ListNode* head, int val){
  if(head == NULL) return head;
  if(head->val == val){
    return deleteNode(head->next, val);
  }
  else{
    head->next = deleteNode(head->next, val);
    return head;
  }
}

void LinkedList::del(int val){
  head = deleteNode(head, val);
}

void LinkedList::print(){
  ListNode* cur = head;
  while(cur!=NULL){
    cout << cur->val << endl;
    cur = cur->next;
  }
}

ListNode* reverseNode(ListNode* head){
  if(head == NULL || head->next == NULL) return head;
  ListNode* newHead = reverseNode(head->next);
  head->next->next = head;
  head->next = NULL;
  
  return newHead;
}

void LinkedList::reverse(){
  head = reverseNode(head);
}

ListNode* mergeTwoSorted(ListNode* left, ListNode* right){
  ListNode* dummy = new ListNode(-1);
  ListNode* cur = dummy, *curL = left, *curR = right;
  while(curL && curR){
    if(curL->val <= curR->val){
      ListNode* temp = new ListNode(curL->val);
      cur->next = temp;
      cur = cur->next;
      curL = curL->next;
    }
    else{
      ListNode* temp = new ListNode(curR->val);
      cur->next = temp;
      cur = cur->next;
      curR = curR->next;
    }
  }
  
  while(curL){
    ListNode* temp = new ListNode(curL->val);
    cur->next = temp;
    cur = cur->next;
    curL = curL->next;
  }
  
  while(curR){
    ListNode* temp = new ListNode(curR->val);
    cur->next = temp;
    cur = cur->next;
    curR = curR->next;
  }
  
  return dummy->next;
}

ListNode* mergeSortLL(ListNode* head){
  if(head == NULL || head->next == NULL) return head;
  
  ListNode* slow = head, *fast = head;
  while(fast->next->next){
    fast = fast->next->next;
    slow = slow->next;
  }
  ListNode* slowNext = slow->next;
  slow->next = NULL;
  ListNode* left = mergeSortLL(head), *right = mergeSortLL(slowNext);
  head = mergeTwoSorted(left, right);
  return head;
}

void LinkedList::sortMerge(){
  head = mergeSortLL(head);
}

void swapNode(ListNode** head_ref, ListNode* min, ListNode* prevMin, ListNode* head){
  *head_ref = min;
  prevMin->next = head;
  
  ListNode* temp = min->next;
  min->next = head->next;
  head->next = temp;
  
}

void changeVal(ListNode* head, int val){
  head->val = val;
}

ListNode* selectionSortLL(ListNode* head){
  if(head == NULL || head->next == NULL) return head;
  ListNode* cur = head;
  ListNode* min = head, *prevMin = NULL;
  while(cur->next){
    if(cur->next->val < min->val){
      min = cur->next;
      prevMin = cur;
    }
    cur = cur->next;
  }

  if(min != head){
    swapNode(&head, min, prevMin, head);
  }

  head->next = selectionSortLL(head->next);
  return head;
}

void LinkedList::sortSelection(){
  head = selectionSortLL(head);
}

int main(){
  LinkedList LL;
  LL.add(3);
  LL.add(6);
  LL.add(5);
  LL.add(2);
  LL.add(1);
  LL.add(7);
  LL.print();
  
  cout << endl;
  
  LL.sortSelection();
  
  LL.print();
  return 0;
}
