
def partition(s: str):
    # possible and unique -> backtracking algorithm

    output = []
    def is_palindrome(word: str) -> bool:
        reverse = word[::-1]
        return word == reverse

    def backtrack(idx, path: str):
        print(path)
        if len(path) == len(s) :
            return

        if len(path) != 0 and is_palindrome(path):
                output.append(path)

        for i in range(0, len(s)):
            if i == idx:
                continue
            new_path = path + s[i]
            backtrack(i, new_path)

        return

    backtrack(-1, "")
    return output
        
print(partition("aab"))
