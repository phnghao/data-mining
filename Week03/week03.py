import unittest

def edit_distance(source_string, target_string):
    m = len(source_string)
    n = len(target_string)

    f = [[0] * (n + 1) for i in range(m + 1)]

    for i in range(1, m + 1):
        f[i][0] = i

    for j in range(1, n + 1):
        f[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if source_string[i-1] == target_string[j-1]:
                a = 0
            else:
                a = 2
            f[i][j] = min(f[i-1][j] + 1, f[i][j-1] + 1, f[i-1][j-1] + a)

    
    operations_performed = []
    i = len(source_string)
    j = len(target_string)

    while (i != 0 and j != 0):
        if source_string[i-1] == target_string[j-1]:
            i -=1
            j -=1

        else:
            if f[i][j] == f[i-1][j-1] + 2:
                operations_performed.append(("Replacement", source_string[i - 1], target_string[j - 1]))
                i -=1 
                j -=1

            elif f[i][j] == f[i-1][j] + 1:
                operations_performed.append(("Delete", source_string[i-1]))
                i -= 1
            
            elif f[i][j] == f[i][j-1] + 1:
                operations_performed.append(("Insert", target_string[j-1]))
                j -= 1

    while(j != 0):
        operations_performed.append(("Insert", target_string[j-1]))
        j -= 1

    while(i != 0):
        operations_performed.append(("Delete", source_string[i-1]))
        i -= 1

    operations_performed.reverse()
    return [f[m][n], operations_performed]

class TestEditDistance(unittest.TestCase):
    
    def test_identical_strings(self):
        dist, ops = edit_distance("hello", "hello")
        self.assertEqual(dist, 0)
        self.assertEqual(len(ops), 0)

    def test_replacement(self):
        dist, ops = edit_distance("a", "b")
        self.assertEqual(dist, 2)
        self.assertEqual(ops[0][0], "Replacement")

    def test_insertion_deletion(self):
        dist, ops = edit_distance("abc", "abcd")
        self.assertEqual(dist, 1)
        self.assertEqual(ops[0][0], "Insert")

    def test_complex_case(self):
        dist, ops = edit_distance("kitten", "sitting")
        self.assertEqual(dist, 5)

def lcs(string1, string2):
    m = len(string1)
    n = len(string2)

    f = [[0] * (n+1) for i in range(m+1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if string1[i-1] == string2[j-1]:
                f[i][j] = f[i-1][j-1] + 1
            else:
                f[i][j] = max(f[i-1][j], f[i][j-1]) 

    operations_performed = []

    i = len(string1)
    j = len(string2)

    while(i != 0 and j != 0):
        if string1[i-1] == string2[j-1]:
            operations_performed.append(string1[i-1])
            i -= 1
            j -= 1
        
        else:
            if f[i-1][j] >= f[i][j-1]:
               i -=1
            else: j -=1

    operations_performed.reverse()
    return [f[m][n], operations_performed]

class TestLCS(unittest.TestCase):

    def test_common_subsequence(self):
        length, seq = lcs("ABCBDAB", "BDCABA")
        self.assertEqual(length, 4)
        self.assertEqual(len(seq), 4)

    def test_no_common(self):
        length, seq = lcs("abc", "def")
        self.assertEqual(length, 0)
        self.assertEqual(seq, [])

    def test_with_empty_string(self):
        length, seq = lcs("", "python")
        self.assertEqual(length, 0)
        self.assertEqual(seq, [])

def dtw(series1, series2):
    INF = float("inf")
    m = len(series1)
    n = len(series2)

    f = [[0]* (n + 1) for i in range(m + 1)]

    for i in range(1, m + 1):
        f[i][0] = INF

    for j in range(1, n + 1):
        f[0][j] = INF

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            d = float(abs(series1[i-1] - series2[j-1]))
            f[i][j] = min(f[i-1][j-1], f[i-1][j], f[i][j-1]) + d

    operations_performed = []
    i = len(series1)
    j = len(series2)

    while(i !=0 and j !=0):
        operations_performed.append(f[i][j])


        if f[i-1][j-1] <= f[i-1][j] and f[i-1][j-1] <= f[i][j-1]:
            i -=1
            j -=1

        elif f[i-1][j] <= f[i][j-1]:
            i -=1
        else:
            j -=1

    while i > 0:
        operations_performed.append(f[i][0])
        i -= 1
    
    while j > 0:
        operations_performed.append(f[0][j])
        j -= 1

    operations_performed.reverse()
    return [f[m][n], operations_performed]

class TestDTW(unittest.TestCase):

    def test_exact_match(self):
        s1 = [1.0, 2.0, 3.0]
        s2 = [1.0, 2.0, 3.0]
        dist, path = dtw(s1, s2)
        self.assertEqual(dist, 0.0)

    def test_warping_capability(self):
        s1 = [1, 2, 3]
        s2 = [1, 1, 2, 2, 3, 3]
        dist, path = dtw(s1, s2)
        self.assertEqual(dist, 0.0)

    def test_different_lengths(self):
        s1 = [1, 5, 8]
        s2 = [1, 9]
        dist, path = dtw(s1, s2)
        self.assertIsInstance(dist, float)
        self.assertTrue(len(path) > 0)
 
def Part1():
    print("# Part 1:")
    print("Enter the source string:")
    source_string = input().strip()
    print("Enter the target string:")
    target_string = input().strip()

    dist, operations_performed = edit_distance(source_string, target_string)

    insertions, deletions, replacements = 0, 0, 0
    for i in operations_performed:
        if i[0] == "Insert":
            insertions += 1
        elif i[0] == "Delete":
            deletions += 1
        else:
            replacements += 1
    print("Results")
    print(f"Minimum edit distance: {dist}")
    print(f"Number of insertions: {insertions}")
    print(f"Number of deletions: {deletions}")
    print(f"Number of replacements: {replacements}")
    print(f"Total number of operations: {insertions + deletions + replacements}")

    print("Actual operations")
    for i in range(len(operations_performed)):
        if operations_performed[i][0] == "Insert":
            print(f"{i+1}) {operations_performed[i][0]} : {operations_performed[i][1]}")
        elif operations_performed[i][0] == "Delete":
            print(f"{i+1}) {operations_performed[i][0]} : {operations_performed[i][1]}")
        else:
            print(f"{i+1}) {operations_performed[i][0]} : {operations_performed[i][1]} by {operations_performed[i][2]}")
    print()

def Part2():
    print("# Part 2:")
    print("Enter the first string")
    string1 = input().strip()
    print("Enter the second string")
    string2 = input().strip()

    lcs_magnitude, operations_performed = lcs(string1, string2)
    print(f"The magnitude of longest common subsequence: {lcs_magnitude}")
    print(f"The possible results: {operations_performed}")
    print()
   
def Part3():
    print("# Part 3:")
    print("Enter the first series")
    line1 = input()
    series1 = [float(x) for x in line1.split()]
    print("Enter the second series")
    line2 = input()
    series2 =[float(x) for x in line2.split()]

    min_dist, operations_performed = dtw(series1, series2)
    print(f"Min distance time warping: {min_dist}")
    operations_performed = list(map(int, operations_performed))
    print(f"The string path warping is: {operations_performed}")
    print()

def main():
    unittest.main(argv = [''], verbosity = 2, exit = False)
    Part1()
    Part2()
    Part3()

if __name__ == "__main__":
    main()