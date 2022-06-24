# AND_gate 함수를 만들어 봅시다.
def AND_gate(x1, x2): # 입력을 두 개 받자, 리턴은 0 or 1
    w1 = 0.4
    w2 = 0.2
    b = -0.5

    w_sum = w1*x1 + w2*x2 + b # 이게 activation func에 들어가야 한다.

    # activation function
    if w_sum >= 0:
        return 1
    else:
        return 0

# AND_gate 호출
print(AND_gate(0, 0)) # 0
print(AND_gate(0, 1)) # 0
print(AND_gate(1, 0)) # 0
print(AND_gate(1, 1)) # 1