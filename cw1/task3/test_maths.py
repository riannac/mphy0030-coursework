#%%
import numpy as np
q = 90
r=90
t=90
rs = np.array([
    [1,0,0,0],
    [0,np.cos(q),np.sin(q),0],
    [0,np.cos(q),(-1)*np.sin(q),0],
    [0,0,0,1]
])
ry = np.array([
    
    [np.cos(r),np.sin(r),0,0],
    [np.cos(r),(-1)*np.sin(r),0,0],
    [0,0,1,0],
    [0,0,0,1]
])
rz = np.array([
    
    [np.cos(t),0,(-1)*np.sin(t),0],
    [0,1,0,0],
    [np.sin(t),0,np.cos(t),0,],
    [0,0,0,1]
])
rot_full = np.zeros_like(rs)
rot_full[1] = rs[1]

# %%
s = 1
q = 1
r = 1
t = 1
s = np.array([[s,0,0,0],
            [0,s,0,0],
            [0,0,s,0],
    [0,0,0,1]])
rz = np.array([
        [1,     0,      0,           0],
        [0,np.cos(q),(-1)*np.sin(q),0],
        [0,np.sin(q),np.cos(q),     0],
        [0,0,0,1]
    ])
rx = np.array([
        [np.cos(r),(-1)*np.sin(r),0,0],
        [np.sin(r), np.cos(r),0,0],
        [0,0,1,0],
        [0,0,0,1]
    ])
ry = np.array([
        [np.cos(t),0,np.sin(t),0],
        [0,1,0,0],
        [(-1)*np.sin(t),0,np.cos(t),0,],
        [0,0,0,1]
    ])

print(np.dot(np.dot(rx,ry),rz))
# %%
sin = (np.sin(2.4))
si = round(np.sin(2.4),6)

def cos(x, rnd = 6):
    co = round(np.cos(x),rnd)
    return co

co = cos(6.24)

def sin(x, rnd = 6):
    si = round(np.sin(x),rnd)
    return si

si = sin(np.pi)
print(co)
#print(sin)
print(si)
# %%
