func twoSum(nums []int, target int) []int {
	m := map[int]int{}

	var diff int

	for i, value := range nums {
		diff = target - value
		// 9 - 2 = 7
		if v, ex := m[value]; ex {
			return []int{v, i}
		} else {
			m[diff] = i
		}
	}

	return []int{}
}