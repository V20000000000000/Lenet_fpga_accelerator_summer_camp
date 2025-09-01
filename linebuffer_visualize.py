# 計算最大數字，並取得其字串長度作為寬度
max_number = 24 * 24 - 1
width = len(str(max_number))

with open("output_aligned.txt", "w") as f:
    for i in range(24):
        for j in range(24):
            number = i * 24 + j
            # 使用 rjust() 將數字字串向右對齊，並在左邊用空格填充至指定寬度
            f.write(str(number).rjust(width) + " ")
        f.write("\n")