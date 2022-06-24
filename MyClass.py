def solution1(shirt_size):
    shirt = ["XS", "S", "M", "L", "XL", "XXL"]
    result = []
    for i in range(len(shirt)):
        cnt = 0
        for j in range(len(shirt_size)):
            if shirt[i] == shirt_size[j]:
                cnt += 1
        result.append(cnt)
    return result

def solution2(arr):
    result = []
    for i in range(len(arr)-1,-1,-1):
        a = arr[i]
        result.append(a)
    return result

def solution3(n):
    result = 0
    for i in range(1, n+1):
        temp = 3*i - 2
        result += temp
    return result

def solution4(n):
    n.sort()
    temp = []
    for i in range(len(n)):
        cnt = 0
        for j in range(len(n)):
            if n[i] == n[j]:
                cnt += 1
        temp.append(cnt)
    temp.sort()
    min = temp[0]
    max = temp[len(temp)-1]
    result = int(max/min)
    return result


# 문제 1번
param1 = ["XS", "S", "L", "L", "XL", "S", "XS"]
ans1 = solution1(param1)
print(ans1)

# 문제 2번

param2 = [1, 4, 2, 3, -999]
ans2 = solution2(param2)
print(ans2)

# 문제 3번
param3 = 5
ans3 = solution3(param3)
print(ans3)

# 문제 4번
param4 = [1, -999, 3, 3, 1, 3, 3, 3, 3, 3, 4, 4, 3]
ans4 = solution4(param4)
print(ans4)
