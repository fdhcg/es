import os


def isLeap(year):
    year=int(year)
    if year%100==0:
        if year%400==0:
            return True
        else:
            return False
    elif year%4==0:
        return True
    else:
        return False 

def timestamp(ts):
    ts=ts.split("T")
    d=ts[0].split("-")
    t=ts[1].split(":")[0]
    if isLeap(d[0]):
        return (mouth2days_leap[int(d[1])]+int(d[2]))*24+int(t)
    else:
        return (mouth2days[int(d[1])]+int(d[2]))*24+int(t)


f=open("./data/data-a-m10.txt","r")
mouth2days=[0,31,59,90,120,153,181,212,243,273,304,334,365]
mouth2days_leap=[0,31,60,91,121,154,182,213,244,274,305,335,366]
data_all=f.readlines()
idx=0
f_w=open("./data/data-m10/{}.txt".format(idx),"w")
for i in range(len(data_all)):
    
    
    f_w.write(data_all[i])
    try:
        t1=timestamp(data_all[i+1].split(",")[0])
        t2=timestamp(data_all[i].split(",")[0])
     
    
    except:
        f_w.close()

    if (t1-t2)!=1:
        f_w.close()
        idx+=1
        f_w=open("data/data-m10/{}.txt".format(idx),"w")
    else:
        continue
    
