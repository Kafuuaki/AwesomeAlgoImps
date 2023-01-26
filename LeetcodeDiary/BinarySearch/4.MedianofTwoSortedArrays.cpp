// 

class Solution {
public:

    int findthekthnumber(vector<int>& nums1, vector<int>& nums2, int kth) {

        // paraphase the problem
        // it asks us to find the kth mininum or the kth and kth - 1 minimum
        // number inside the two array
        // with binary seach, we first find the number with kth - 1 / 2 index
        // in both array (we need to consider the boundary later)

        // we need to maintain two pointers so that we can search in the two array

        int n1_size = nums1.size();
        int n2_size = nums2.size();

        // the two pointer we need to maintain
        int p1 = 0, p2 = 0;

        while(true) {

            // exlude all elements in one array
            if (p1 == n1_size) {
                return nums2[p2 + kth - 1];
            }
            if (p2 == n2_size) {
                return nums1[p1 + kth - 1];
            }

            if (kth == 1) {
                return min(nums1[p1], nums2[p2]);
            }

            int new_p1 = min(p1 + kth / 2 - 1, n1_size - 1);
            int new_p2 = min(p2 + kth / 2 - 1, n2_size - 1);
            int pivot1 = nums1[new_p1];
            int pivot2 = nums2[new_p2];

            if(pivot1 <= pivot2) {
                kth -= new_p1 - p1 + 1;
                p1 = new_p1 + 1;
            } else {
                kth -= new_p2 - p2 + 1;
                p2 = new_p2 + 1;
            }
        }
    }


    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        // the things needed to be taken into consideration
        // we need to finish it in O(log(m+n)) -> likely binary search
        // the two arrays are sorted
        // m + n is even or odd number -> if m + n is even, the median is
        // m + n / 2, else the median is m + n + 1 / 2
        // the median number is in nums1 or nums2
        // the size of length 1 and length 2
        // how to deal with index


        // there is different to divided by 2 and 2.0
        int len_n1n2 = nums1.size() + nums2.size();
        if (len_n1n2 % 2) {
            return findthekthnumber(nums1, nums2, (len_n1n2 + 1) / 2);
        } else {
            return (findthekthnumber(nums1, nums2, len_n1n2 / 2) +
            findthekthnumber(nums1, nums2, len_n1n2 / 2 + 1)) / 2.0;
        }

        return 0;
    }
};