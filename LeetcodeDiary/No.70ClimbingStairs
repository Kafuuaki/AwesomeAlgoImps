class Solution {
public:
    int dp[50];

    int climbStairs(int n) {
        // when n == 1, there is only 1 way (1)
        // when n == 2, there could be 1 + 1, or 2 (2)
        // when n == 3, there could be 1 + 1 + 1 or 1 + 2 or 2 + 1 (3)
        // when n == 4, there could be 1 + 1 + 1 + 1 or 1 + 1 + 2 or 1 + 2 + 1
        // or 2 + 1 + 1 or 2 + 2 (5)
        // when n is even, n == 2c for some integer c
        // when n is odd, n == 2c + 1 for some integer c

        // wow.. the final state can be determined by the previous state
        // since for the final step at height h, one coule take one step from h - 1
        // or two step from h - 2
        // therefore, the final state in determined by the state h - 1 and h - 2
        // f(x) = f (x - 1) + f(x - 2)
        
        // base case
        if (n == 1) return 1;
        if (n == 2) return 2;

        dp[1] = 1, dp[2] = 2; 
        for (int i = 3; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }

        return dp[n];
    }
};