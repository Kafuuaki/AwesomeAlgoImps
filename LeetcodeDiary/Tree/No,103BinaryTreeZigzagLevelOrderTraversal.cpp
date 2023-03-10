// bfs version https://leetcode.cn/problems/binary-tree-right-side-view/solution/er-cha-shu-de-you-shi-tu-by-leetcode-solution/

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
    vector<vector<int>> sol;

    void bfs(TreeNode *root)
    {
        queue<TreeNode *> q;
        // indicates that the level is even of not, if the level is even,
        // we count from right to left
        int level = 1;
        q.push(root);

        while (!q.empty())
        {
            // using deque to maintain the order of output
            // clear the output for every layer
            deque<int> d;

            // iterate every node that saved in the queue in the layer
            int len_q = q.size();
            for (int i = 0; i < len_q; i++)
            {
                auto node = q.front();
                q.pop();

                // even or odd number
                if (level % 2)
                {
                    d.push_back(node->val);
                }
                else
                {
                    d.push_front(node->val);
                }

                if (node->left)
                {
                    q.push(node->left);
                }

                if (node->right)
                {
                    q.push(node->right);
                }
            }

            sol.emplace_back(vector<int>(d.begin(), d.end()));
            level++;
        }
    }

    vector<vector<int>> zigzagLevelOrder(TreeNode *root)
    {
        // let us consider the case given
        // by inspiration, the task can be solved by bfs
        // we need to distinguish the odd level and even level of the tree

        if (root == nullptr)
            return sol;
        bfs(root);
        return sol;
    }
};

// dfs version
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
    vector<int> sol;

    void dfs(TreeNode *root, int level)
    {
        if (!root)
            return;

        // when to push
        // when dfs visit the first deeper node
        // the size of solution is the current depth iterated
        if (level == sol.size())
        {
            sol.push_back(root->val);
        }

        // for the next node, the level is current level + 1
        level++;
        // dfs recursion
        // always go right first
        dfs(root->right, level);
        dfs(root->left, level);
    }

    vector<int> rightSideView(TreeNode *root)
    {
        dfs(root, 0);
        return sol;
    }
};