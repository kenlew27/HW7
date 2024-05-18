import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

arr = np.loadtxt("LAS_18258650.las.csv", delimiter=" ", dtype=str)
arr = arr[arr[:, 0].argsort()]
tarr = arr.transpose().astype(int)


omaxx = np.max(tarr[0])
ominx = np.min(tarr[0])
omeanx = np.mean(tarr[0])
ostdx = np.std(tarr[0])

omaxy = np.max(tarr[1])
ominy = np.min(tarr[1])
omeany = np.mean(tarr[1])
ostdy = np.std(tarr[1])

omaxz = np.max(tarr[2])
ominz = np.min(tarr[2])
omeanz = np.mean(tarr[2])
ostdz = np.std(tarr[2])

omaxi = np.max(tarr[3])
omini = np.min(tarr[3])
omeani = np.mean(tarr[3])
ostdi = np.std(tarr[3])

np.random.shuffle(arr)
rantarr = arr[:1000000].transpose().astype(int)


milmaxx = np.max(rantarr[0])
milminx = np.min(rantarr[0])
milmeanx = np.mean(rantarr[0])
milstdx = np.std(rantarr[0])

milmaxy = np.max(rantarr[1])
milminy = np.min(rantarr[1])
milmeany = np.mean(rantarr[1])
milstdy = np.std(rantarr[1])

milmaxz = np.max(rantarr[2])
milminz = np.min(rantarr[2])
milmeanz = np.mean(rantarr[2])
milstdz = np.std(rantarr[2])

milmaxi = np.max(rantarr[3])
milmini = np.min(rantarr[3])
milmeani = np.mean(rantarr[3])
milstdi = np.std(rantarr[3])







#1
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

df = pd.DataFrame(columns=['Observations', 'Original Data', '1 Million Data'])
df['Observations'] = ['Max X', 'Min X', 'Mean X','STD X',    'Max Y', 'Min Y', 'Mean Y','STD Y',        'Max Z', 'Min Z', 'Mean Z','STD Z', 'Max I', 'Min I', 'Mean I', 'STD I' ]
df['Original Data'] = [ omaxx , ominx , omeanx, ostdx ,     omaxy , ominy , omeany, ostdy ,             omaxz , ominz , omeanz, ostdz ,       omaxi , omini, omeani, ostdi                     ]
df['1 Million Data'] = [ milmaxx , milminx , milmeanx, milstdx ,     milmaxy , milminy , milmeany, milstdy ,             milmaxz , milminz , milmeanz, milstdz ,       milmaxi , milmini, milmeani, milstdi                     ]
print(df)








#2
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

print("correlation between x and y: ", np.corrcoef(tarr[0],tarr[1])[0][1])
print("correlation between x and z: ",np.corrcoef( tarr[0],tarr[2])[0][1])
print("correlation between y and z: ",np.corrcoef( tarr[1],tarr[2])[0][1])

print("correlation between x and intensity: ",np.corrcoef( tarr[0],tarr[3])[0][1])
print("correlation between y and intensity: ",np.corrcoef( tarr[1],tarr[3])[0][1])
print("correlation between z and intensity: ",np.corrcoef( tarr[2],tarr[3])[0][1])










#3
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

xhist = plt.hist(tarr[0], bins='auto')
plt.title("X Distribution")
plt.show()
yhist = plt.hist(tarr[1], bins='auto')
plt.title("Y Distribution")
plt.show()
zhist = plt.hist(tarr[2], bins='auto')
plt.title("Z Distribution")
plt.show()
ihist = plt.hist(tarr[3], bins='auto')
plt.title("Intensity Distribution")
plt.show()





#4 
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#Z VAL MEAN
zi, yi, xi = np.histogram2d(tarr[1], tarr[0], bins=(2500,2500), weights=tarr[2], density=False)
counts, _, _ = np.histogram2d(tarr[1], tarr[0], bins=(2500,2500))
counts[counts == 0] = 1
zi = zi / counts
zi = np.ma.masked_invalid(zi)
fig, ax = plt.subplots()
fig, ax = plt.subplots()
ax.pcolormesh(xi, yi, zi)
plt.show()



#INTENSITY VAL MEAN
zi, yi, xi = np.histogram2d(tarr[1], tarr[0], bins=(2500,2500), weights=tarr[3], density=False)
counts, _, _ = np.histogram2d(tarr[1], tarr[0], bins=(2500,2500))
counts[counts == 0] = 1
zi = zi / counts
zi = np.ma.masked_invalid(zi)
fig, ax = plt.subplots()

fig, ax = plt.subplots()
ax.pcolormesh(xi, yi, zi)

plt.show()



#6
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



#INTENSITY VAL MEAN
zi, yi, xi = np.histogram2d(tarr[1], tarr[0], bins=(2500,2500), weights=tarr[3], density=False)

prevx= 0
nxtx= 0
prevy= 0
nxty= 0
average = 0

for i, val in enumerate(zi):
    for j, val2 in enumerate(val):
        if i==0:
            prevy = 0
        if i!=0:
            prevy = zi[i-1][j]
        if j==0:
            prevx=0
        if j!=0:
            prevx=zi[i][j-1]

        if i==len(val)-1:
            nxty = 0
        if i!=len(val)-1:
            nxty = zi[i+1][j]
        if j==len(val)-1:
            nxty = 0
        if j!=len(val)-1:
            nxty = zi[i][j+1]
        if zi[i][j] == 0:
            if prevx == 0:
                average = (prevy+nxtx+nxty)/3
            if prevy == 0:
                average = (prevx+nxtx+nxty)/3
            if nxtx == 0:
                average = (prevx+prevy+nxty)/3
            if nxty == 0:
                average = (prevx+prevy+nxtx)/3
            else:
                average = (prevx+prevy+nxtx+nxty)/4
            zi[i][j] = average

counts, _, _ = np.histogram2d(tarr[1], tarr[0], bins=(2500,2500))
counts[counts == 0] = 1
zi = zi / counts
zi = np.ma.masked_invalid(zi)
fig, ax = plt.subplots()

fig, ax = plt.subplots()
ax.pcolormesh(xi, yi, zi)

plt.show()







#Z VAL MEAN
zi, yi, xi = np.histogram2d(tarr[1], tarr[0], bins=(2500,2500), weights=tarr[2], density=False)

prevx= 0
nxtx= 0
prevy= 0
nxty= 0
average = 0

for i, val in enumerate(zi):
    for j, val2 in enumerate(val):
        if i==0:
            prevy = 0
        if i!=0:
            prevy = zi[i-1][j]
        if j==0:
            prevx=0
        if j!=0:
            prevx=zi[i][j-1]

        if i==len(val)-1:
            nxty = 0
        if i!=len(val)-1:
            nxty = zi[i+1][j]
        if j==len(val)-1:
            nxty = 0
        if j!=len(val)-1:
            nxty = zi[i][j+1]
        if zi[i][j] == 0:
            if prevx == 0:
                average = (prevy+nxtx+nxty)/3
            if prevy == 0:
                average = (prevx+nxtx+nxty)/3
            if nxtx == 0:
                average = (prevx+prevy+nxty)/3
            if nxty == 0:
                average = (prevx+prevy+nxtx)/3
            else:
                average = (prevx+prevy+nxtx+nxty)/4
            zi[i][j] = average *0.1

counts, _, _ = np.histogram2d(tarr[1], tarr[0], bins=(2500,2500))
counts[counts == 0] = 1
zi = zi / counts
zi = np.ma.masked_invalid(zi)
fig, ax = plt.subplots()

fig, ax = plt.subplots()
ax.pcolormesh(xi, yi, zi)

plt.show()








#8
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#Intensity col Variance Constant
zi, yi, xi = np.histogram2d(tarr[1], tarr[0], bins=(2500,2500), weights=tarr[3], density=False)

varsum =0
varav =0
for val in zi:
    varsum+=np.var(val)
varav = varsum / len(zi)

tol = 0.01*varav


for i, val in enumerate(zi):
    val=np.array(val)
    difference = varav - np.var(val)

    while (abs(difference) > tol):
        print('cur var:', np.var(val), ' target var:', varav, 'iter',i)
        if difference <0:
            val = val * 0.999
            difference = varav - np.var(val)
        if difference >0:
            val = val  * 1.002
            difference = varav - np.var(val)
   
    zi[i] =val

counts, _, _ = np.histogram2d(tarr[1], tarr[0], bins=(2500,2500))
counts[counts == 0] = 1
zi = zi / counts
zi = np.ma.masked_invalid(zi)
fig, ax = plt.subplots()

fig, ax = plt.subplots()
ax.pcolormesh(xi, yi, zi)

plt.show()





#Intensity row Variance Constant

zi, yi, xi = np.histogram2d(tarr[1], tarr[0], bins=(2500,2500), weights=tarr[3], density=False)

zi = zi.transpose()

varsum =0
varav =0
for val in zi:
    varsum+=np.var(val)
varav = varsum / len(zi)

tol = 0.01*varav


for i, val in enumerate(zi):
    val=np.array(val)
    difference = varav - np.var(val)

    while (abs(difference) > tol):
        print('cur var:', np.var(val), ' target var:', varav, 'iter',i)
        if difference <0:
            val = val * 0.999
            difference = varav - np.var(val)
        if difference >0:
            val = val  * 1.002
            difference = varav - np.var(val)
   
    zi[i] =val

zi = zi.transpose()

counts, _, _ = np.histogram2d(tarr[1], tarr[0], bins=(2500,2500))
counts[counts == 0] = 1
zi = zi / counts
zi = np.ma.masked_invalid(zi)
fig, ax = plt.subplots()

fig, ax = plt.subplots()
ax.pcolormesh(xi, yi, zi)

plt.show()




#Both


zi, yi, xi = np.histogram2d(tarr[1], tarr[0], bins=(2500,2500), weights=tarr[3], density=False)

varsum =0
varav =0
for val in zi:
    varsum+=np.var(val)
varav = varsum / len(zi)

tol = 0.01*varav


for i, val in enumerate(zi):
    val=np.array(val)
    difference = varav - np.var(val)

    while (abs(difference) > tol):
        print('cur var:', np.var(val), ' target var:', varav, 'iter',i)
        if difference <0:
            val = val * 0.999
            difference = varav - np.var(val)
        if difference >0:
            val = val  * 1.002
            difference = varav - np.var(val)
   
    zi[i] =val

zi = zi.transpose()

varsum =0
varav =0
for val in zi:
    varsum+=np.var(val)
varav = varsum / len(zi)

tol = 0.01*varav


for i, val in enumerate(zi):
    val=np.array(val)
    difference = varav - np.var(val)

    while (abs(difference) > tol):
        print('cur var:', np.var(val), ' target var:', varav, 'iter',i)
        if difference <0:
            val = val * 0.999
            difference = varav - np.var(val)
        if difference >0:
            val = val  * 1.002
            difference = varav - np.var(val)
   
    zi[i] =val

zi = zi.transpose()

counts, _, _ = np.histogram2d(tarr[1], tarr[0], bins=(2500,2500))
counts[counts == 0] = 1
zi = zi / counts
zi = np.ma.masked_invalid(zi)
fig, ax = plt.subplots()

fig, ax = plt.subplots()
ax.pcolormesh(xi, yi, zi)

plt.show()







#9 ROI
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



startx=500
endx=1000
starty=500
endy=1000
binx = endx - startx
biny = endy - starty

zi, yi, xi = np.histogram2d(tarr[1], tarr[0], bins=(2500,2500), weights=tarr[3], density=False)

xi = xi[startx:endx]
yi = yi[starty:endy]


zi = zi[starty:endy]
zi = zi.transpose()
zi = zi[startx:endx]
zi = zi.transpose()

prevx= 0
nxtx= 0
prevy= 0
nxty= 0
average = 0

for i, val in enumerate(zi):
    for j, val2 in enumerate(val):
        if i==0:
            prevy = 0
        if i!=0:
            prevy = zi[i-1][j]
        if j==0:
            prevx=0
        if j!=0:
            prevx=zi[i][j-1]

        if i==len(val)-1:
            nxty = 0
        if i!=len(val)-1:
            nxty = zi[i+1][j]
        if j==len(val)-1:
            nxty = 0
        if j!=len(val)-1:
            nxty = zi[i][j+1]
        if zi[i][j] == 0:
            if prevx == 0:
                average = (prevy+nxtx+nxty)/3
            if prevy == 0:
                average = (prevx+nxtx+nxty)/3
            if nxtx == 0:
                average = (prevx+prevy+nxty)/3
            if nxty == 0:
                average = (prevx+prevy+nxtx)/3
            else:
                average = (prevx+prevy+nxtx+nxty)/4
            zi[i][j] = average

counts, _, _ = np.histogram2d(tarr[1], tarr[0], bins=(2500,2500))

counts = counts[starty:endy]
counts = counts.transpose()
counts = counts[startx:endx]
counts = counts.transpose()

counts[counts == 0] = 1
zi = zi / counts
zi = np.ma.masked_invalid(zi)
fig, ax = plt.subplots()

fig, ax = plt.subplots()
ax.pcolormesh(xi, yi, zi)

plt.show()

