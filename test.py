a = [1, 2, 3]
b = [4, 5, 6]

temp = [a, b]

for i in temp:
    print(i)
    i[0] = 9

print(temp)
print(a, b) # 세상에 정말 원본이 바뀌잖아? 놀랍다.