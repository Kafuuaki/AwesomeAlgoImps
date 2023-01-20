// bfs solution

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution
{
public:
    vector<int> sol = {};

    void bfs(TreeNode *root)
    {
        // queue to maintain the nodes needed to be search
        queue<TreeNode *> q;

        // insantiate the queue
        q.push(root);

        // iteration starts
        while (!q.empty())
        {

            // iteration times, which is the number of all nodes currently in queue
            int q_size = q.size();

            // iterate all the nodes
            // when to push to sol
            for (int i = 0; i < q_size; i++)
            {
                // take the nodes out
                auto node = q.front();
                q.pop();

                // be careful to visit the left node first, since we need the rightest node be visited last
                if (node->left)
                    q.push(node->left);
                if (node->right)
                    q.push(node->right);

                // always push the rightest one in the deepest iterated level
                if (i == q_size - 1)
                    sol.push_back(node->val);
            }
        }
    }

    vector<int> rightSideView(TreeNode *root)
    {
        if (!root)
            return {};

        bfs(root);

        return sol;
    }
};

// dfs solution