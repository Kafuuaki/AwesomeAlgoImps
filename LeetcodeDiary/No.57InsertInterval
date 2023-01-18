class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        int len_intervals = intervals.size();

        if (len_intervals == 0) return {};

        // sort the given sets of intervals by lower bound of every interval
        sort(intervals.begin(), intervals.end());

        vector<vector<int>> sol;

        // push the first interval in
        sol.push_back(intervals[0]);

        // replace the upper bound of the array when the overlapping intervals have 
        // greater upper bound
        for (int i = 1; i < len_intervals; i++) {
            // be careful about the case like [[1, 4], [1, 4]], which may cause that
            // same situation

            // overalpping situation
            // bounds check
            if (sol.back()[1] >= intervals[i][0]){
                // the upper bound outside the iterval 
                if (sol.back()[1] <= intervals[i][1]) {
                    sol.back()[1] = intervals[i][1];
                }
                // the upper bound inside the interval
            } else {
                // not cross
                    sol.push_back(intervals[i]);
            }
        }

        return sol;
    }
};