# -*- coding: utf-8 -*
#!/usr/bin/env python

def LCS_length_3(x, y):
    xlen = len(x);
    ylen = len(y);
    tmp_1 = [[0 for i in range(ylen + 1)] for j in range(2)]
    tmp_2 = [[0 for i in range(ylen + 1)] for j in range(xlen + 1)]
    for i in range(1, xlen+1):
        for j in range(1, ylen+1):
            if(x[i-1] == y[j-1]):
                tmp_1[i%2][j] = tmp_1[(i-1)%2][j-1] + 1;
                tmp_2[i][j] = 0
            else:
                if(tmp_1[i%2][j-1] >= tmp_1[(i-1)%2][j]):
                    tmp_1[i%2][j] = tmp_1[i%2][j-1]
                    tmp_2[i][j] = 1
                else:
                    tmp_1[i%2][j] = tmp_1[(i-1)%2][j]
                    tmp_2[i][j] = -1
    return tmp_1, tmp_2

def LCS_print(x, y, tmp_2):
    result = []
    i = len(x)
    j = len(y)
    k = 0
    while(i > 0 and j > 0):
        if(tmp_2[i][j] == 0):
            result.append(x[i-1])
            k = k + 1
            i = i - 1
            j = j - 1
        elif tmp_2[i][j] == 1:
            j = j - 1
        elif tmp_2[i][j] == -1:
            i = i - 1
    return result
def get_sim(sent1,sent2):
    max_len, tmp_2 = LCS_length_3(sent1, sent2)
    lens=LCS_print(sent1,sent2,tmp_2)
    sim=len(lens)*2/(len(sent1)+len(sent2))
    return sim
if __name__ == '__main__':
    sent1 = "iloveyou"
    sent2 = "youlovei"
    print(get_sim(sent1, sent2))