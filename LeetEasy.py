import math


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    # Q122 168 is tricky

    # Q1
    def twoSum(self, nums: [int], target: int) -> [int]:
        for i in range(0, len(nums)):
            a = target - nums[i]
            for j in range(0, len(nums)):
                if j == i:
                    continue
                if a == nums[j]:
                    return [i, j]

    # def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:

    # Q27
    def removeElement(self, nums: [int], val: int) -> int:
        counter = 0
        length = len(nums)
        i = 0
        while i < len(nums):
            if nums[i] != val:
                counter = counter + 1
            else:
                del (nums[i])
                i = i - 1
            if len(nums) == i - 2:
                break
            i += 1
        return counter

    #  Q28
    def strStr(self, haystack: str, needle: str) -> int:
        flag = 1
        if len(needle) > len(haystack):
            return -1
        if needle == "":
            return 0
        difference = len(haystack) - len(needle)
        for i in range(0, difference + 1):
            temp_i = i
            for j in range(0, len(needle)):
                if i + j >= len(haystack):
                    flag = 0
                    break
                if haystack[i + j] != needle[j]:
                    flag = 0
                    break
            if flag == 1:
                return temp_i
            flag = 1
        return -1

    # Q35
    def searchInsert(self, nums: [int], target: int) -> int:
        for i in range(0, len(nums)):
            if target <= nums[i]:
                return i
        return i + 1

    # Pre-requisite for Q38.
    def divideString(self, s: str) -> [[int], [int]]:
        store = [[], []]
        newnum = 0
        i = 0
        if s == "":
            return [[], []]
        store[0].append(0)
        store[1].append(1)
        store[0][newnum] = int(s[i])
        while i < len(s) - 1:
            if s[i] == s[i + 1]:
                store[1][newnum] = store[1][newnum] + 1
            else:
                store[0].append(0)
                store[1].append(1)
                newnum = newnum + 1
                store[0][newnum] = int(s[i + 1])
            i = i + 1
        return store

    # Second Pre-requisite for Q38.
    def countAndSay_implem(self, store: [[int], [int]]) -> str:
        stri = ""
        for j in range(0, len(store[0])):
            stri = stri + str(store[1][j]) + str(store[0][j])
        return stri

    #  Q38
    def countAndSay(self, n: int) -> str:
        if n == 1:
            return "1"
        return Solution.countAndSay_implem(self, Solution.divideString(self, Solution.countAndSay(self, n - 1)))

    # Q53
    def maxSubArray(self, nums: [int]) -> int:
        dic = []
        dic.append(nums[0])
        for i in range(1, len(nums)):
            if dic[i - 1] <= 0:
                dic.append(nums[i])
            else:
                dic.append(dic[i - 1] + nums[i])
        return max(dic)

    # Q58
    def lengthOfLastWord(self, s: str) -> int:
        leng = len(s)
        flag = 0
        if leng == 0:
            return 0
        if s[leng - 1] != " ":
            for i in range(0, leng):
                if s[leng - 1 - i] == " ":
                    start = leng - 1 - i
                    return leng - start - 1
            return leng
        else:
            for i in range(0, leng):
                if flag == 0:
                    if s[leng - 1 - i] != " ":
                        end = leng - 1 - i
                        flag = 1
                        continue
                if flag == 1:
                    if s[leng - 1 - i] == " ":
                        return end - (leng - 1 - i)
            if flag == 0:
                return 0
            else:
                return end + 1

    # Q66
    def plusOne(self, digits: [int]) -> [int]:
        length = len(digits)
        # flag[] control whether current digit should +1 or not
        flag = [0] * length
        flag[length - 1] = 1
        if digits == [9] * length:
            return [1] + [0] * length
        for i in range(0, length):
            if flag[length - 1 - i] == 1:
                if digits[length - 1 - i] == 9:
                    flag[length - 2 - i] = 1
                    digits[length - 1 - i] = 0
                else:
                    digits[length - 1 - i] += 1
        return digits

    # Q67
    def addBinary(self, a: str, b: str) -> str:
        length_a, length_b = len(a), len(b)
        length = max(length_a, length_b)
        new_a = '0' * (length - length_a) + a
        new_b = '0' * (length - length_b) + b
        # flag[] control whether current digit should +1 or not.
        # There's an extra element in flag[] than in new_a/b,
        # controlling the +1 on the most significant digit.
        flag = [0] * (length + 1)
        flag[length] = 0
        result = ''
        for i in range(0, length):
            current = int(new_a[length - 1 - i]) + int(new_b[length - 1 - i])
            if flag[length - i]:
                if current == 0:
                    result += '1'
                elif current == 1:
                    result += '0'
                    flag[length - 1 - i] = 1
                else:
                    result += '1'
                    flag[length - 1 - i] = 1
            else:
                if current == 0:
                    result += '0'
                elif current == 1:
                    result += '1'
                else:
                    result += '0'
                    flag[length - 1 - i] = 1
        if flag[0] == 1:
            result += '1'
        return result[-1::-1]

    # Q69
    # Binary search in O(log n).
    def mySqrt(self, x: int) -> int:
        lower, upper = 0, x
        while (lower <= upper):
            mid = (lower + upper) // 2
            square = mid ** 2
            if (square == x):
                return mid
            if (square < x):
                lower = mid + 1
            else:
                upper = mid - 1
        return upper

    # Q70
    def climbStairs_recursion(self, n: int) -> int:
        if n == 1:
            return 1
        if n == 2:
            return 2
        return Solution.climbStairs(self, n - 1) + Solution.climbStairs(self, n - 2)

    def climbStairs(self, n: int) -> int:
        a = 1
        b = 1
        while n >= 1:
            temp = a
            a = b
            b = temp + b
            n = n - 1
        return a

    # Q100 Not Finished
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, x):
    #         self.val = x
    #         self.left = None
    #         self.right = None

    # def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
    # p.left = asd

    # Q119
    # Also the pre-requisite for function "generate" (Q118)
    # Warning:
    # Should change the starting index below and the function name for solution of Q119!
    def pascal_oneline(self, numRow: int) -> [int]:
        current = [1]
        if numRow == 1:
            return [1]
        if numRow == 2:
            return [1, 1]
        prev = self.pascal_oneline(numRow - 1)
        for i in range(0, len(prev) - 1):
            current.append(prev[i] + prev[i + 1])
        current.append(1)
        return current

    # Q118 Not the optimal choice. Wasting time calculating repeating results.
    def generate(self, numRows: int) -> [[int]]:
        result = []
        for i in range(1, numRows + 1):
            result.append(Solution.pascal_oneline(self, i))
        return result

    #  Q121 solution in O(n)
    def maxProfit(self, prices: [int]) -> int:
        if len(prices) == 1 or len(prices) == 0:
            return 0
        buy = prices[0]
        poten_prof = 0
        for i in range(1, len(prices)):
            poten_prof = max(poten_prof, prices[i] - buy)
            if prices[i] < buy:
                buy = prices[i]
        return poten_prof

    # Q122 Solution
    def maxProfit_Q122(self, prices: [int]) -> int:
        res = 0
        for i in range(len(prices) - 1):
            p = prices[i + 1] - prices[i]
            if p > 0:
                res += p
        return res

    # Q125
    def isPalindrome(self, s: str) -> bool:
        low = s.lower()
        alphanumeric = ''
        for i in low:
            if i.isalnum():
                alphanumeric += i
        length = len(alphanumeric)
        if length == 0 or length == 1:
            return True
        for j in range(0, length):
            if alphanumeric[j] != alphanumeric[length - j - 1]:
                return False
        return True

    # Q136 XOR in time O(n), space O(1)
    def singleNumber(self, nums: [int]) -> int:
        res = 0
        for i in nums:
            res ^= i
        return res

    # Q136 Hash table (dictionary) in time O(n), space O(n)
    def singleNumber(self, nums):
        hash_table = {}
        for i in nums:
            try:
                hash_table.pop(i)
            except:
                hash_table[i] = 1
        return hash_table.popitem()[0]

    # Q167 TwoSum 
    def twoSum_2(self, numbers: [int], target: int) -> [int]:
        length = len(numbers)
        for i in range(0, length):
            if numbers[i] == target / 2:
                if numbers[i + 1] == numbers[i]:
                    return [i + 1, i + 2]
            else:
                for j in range(i + 1, length):
                    if numbers[i] == numbers[j]:
                        break
                    if numbers[i] + numbers[j] == target:
                        return [i + 1, j + 1]

    # Q168
    def convertToTitle(self, num):
        result = ""
        while num > 0:
            if num % 26 == 0:
                result = "Z" + result
                num = int(num / 26) - 1
            else:
                result = chr((num % 26) + 64) + result
                num = int(num / 26)
        return result

    # Q169
    # dic.items() returns a dic_items object,
    # where dictionary items can be scanned by
    #  "for key, val in dic_items()"
    def majorityElement(self, nums: [int]) -> int:
        dic = {}
        length = len(nums)
        for i in range(0, length):
            if nums[i] not in dic:
                dic[nums[i]] = 1
            else:
                dic[nums[i]] += 1
        for key, vall in dic.items():
            if vall >= length / 2:
                return key

    def test(self, li):
        for a, b in li:
            print(b)

    # Q171
    # orc(str) -> its ASCII code (int)
    # chr(int) -> corresponding character of int in ASCII table
    def titleToNumber(self, s: str) -> int:
        length = len(s)
        flag = [0] * (length + 1)
        nums = [0] * (length + 1)
        for i in range(0, length):
            if flag[length - i] == 0:
                if s[length - i - 1] == 'Z':
                    nums[length - i] = 0
                    flag[length - i - 1] = 1
                else:
                    nums[length - i] = ord(s[length - 1 - i]) - 64
            else:
                if s[length - i - 1] == 'Y':
                    nums[length - i] = 0
                    flag[length - i - 1] = 1
                elif s[length - i - 1] == 'Z':
                    nums[length - i] = 1
                    flag[length - i - 1] = 1
                else:
                    nums[length - i] = ord(s[length - 1 - i]) + 1 - 64
        if flag[0] == 1:
            nums[0] = 1
        result = 0
        length2 = len(nums)
        for j in range(0, length2):
            result += nums[j] * (26 ** (length2 - j - 1))
        return result

    # Q172
    # Good explaination: 
    # https://leetcode.com/problems/factorial-trailing-zeroes/...
    # discuss/52424/Iterative-Python-solution-WITH-EXPLANATION
    def trailingZeroes(self, n):
        k, tot = 5, 0
        while k <= n:
            tot += n // k
            k = k * 5
        return tot

    # Q189 in time O(n) space O(1) using cyclic replacement
    def rotate(self, nums: [int], k: int) -> None:
        length = len(nums)
        repeat = 0
        gcd_val = math.gcd(length, k)
        while repeat < gcd_val:
            to_store = 0
            to_insert = nums[repeat]
            counter = 0
            switch_index = repeat
            repeat += 1
            while counter < length / gcd_val:
                to_store = nums[(switch_index + k) % length]
                nums[(switch_index + k) % length] = to_insert
                switch_index = (switch_index + k) % length
                counter += 1
                to_insert = to_store
        print(nums)

    # Q190
    def reverseBits(self, n: int) -> int:
        return int(str(bin(n))[2:].zfill(32)[::-1], 2)

    # Qucik sort
    def quicksort(self, arr: [int]) -> [int]:    
        if len(arr) <= 1:        
            return arr    
        pivot = arr[len(arr) // 2]    
        left = [x for x in arr if x < pivot]    
        middle = [x for x in arr if x == pivot]    
        right = [x for x in arr if x > pivot]    
        return Solution.quicksort(self, left) + middle + Solution.quicksort(self, right)

a = Solution()
print(a.quicksort([3,5,7,1,2,4]))