import os
os.getcwd()

def dircheck(AF, p = '.'):
    A = os.listdir(p)
    for d in A:
        try:
            os.chdir(d)
            AF = dircheck(AF, '.')
            PF = os.listdir('.')
            PF = [os.getcwd() + '/' + pf for pf in PF]
            AF.extend(PF)
            os.chdir('..')
        except:
            pass
    return AF

AF = []
AF = dircheck(AF)
AF.sort()
for af in AF:
    print(af)