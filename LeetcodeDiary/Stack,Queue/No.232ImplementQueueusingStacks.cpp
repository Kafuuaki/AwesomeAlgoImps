class MyQueue
{
private:
    // The elements in queue first in, first out; last in, last out
    // The elements in the stack first in, last out;last in, first out
    // Thus to push an element in the queue, the element is the same as
    // the element of stack in reversed the order.
    // Thus we need a order to push
    stack<int> s_order, r_order;

    void s_to_r()
    {
        while (!s_order.empty())
        {
            r_order.push(s_order.top());
            s_order.pop();
        }
    }

public:
    MyQueue() {}

    void push(int x)
    {
        // since this is only the push, just push it into the s_stack
        s_order.push(x);
    }

    int pop()
    {
        // if there is an element in r_order, we cannot directly put the element
        // in the r_order
        if (r_order.empty())
        {
            s_to_r();
        }

        int x = r_order.top();
        r_order.pop();
        return x;
    }

    int peek()
    {
        if (r_order.empty())
        {
            s_to_r();
        }
        return r_order.top();
    }

    bool empty()
    {
        if (s_order.empty() && r_order.empty())
        {
            return true;
        }

        return false;
    }
};

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue* obj = new MyQueue();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->peek();
 * bool param_4 = obj->empty();
 */