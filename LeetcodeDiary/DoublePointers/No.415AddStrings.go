// go

import (
	"fmt"
	"strconv"
)

func addStrings(num1 string, num2 string) string {
	sol := ""
	add := 0
	for i, j := len(num1)-1, len(num2)-1; i >= 0 || j >= 0 || add != 0; i, j = i-1, j-1 {
		var x, y int
		if i >= 0 {
			x = int(num1[i] - '0')
		}

		if j >= 0 {
			y = int(num2[j] - '0')
		}

		cur := x + y + add
		sol = strconv.Itoa(cur%10) + sol
		add = cur / 10
	}

	return sol
}