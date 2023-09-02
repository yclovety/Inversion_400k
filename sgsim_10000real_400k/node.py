import math
import numpy as np

class NODE:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0

class CELL:
    def __init__(self): # 不加self就变成了对所有类对象同时更改
        self.vertices = [-1, -1, -1, -1, -1, -1, -1, -1]
        self.neighbors = [-1, -1, -1, -1, -1, -1]
        self.dx = 0
        self.dy = 0
        self.dz = 0
        self.volume = 0
        self.xc = 0
        self.yc = 0
        self.zc = 0
        self.porosity = 0
        self.kx = 0
        self.ky = 0
        self.kz = 0
        self.trans = [0, 0, 0, 0, 0, 0]
        self.markbc = 0
        self.markwell = 0
        self.press = 0

print("build Grid")
dxvec=[0]
for i in range(0, 20):
    dxvec.append(5)

dyvec=[0]
for i in range(0, 20):
    dyvec.append(5)
dzvec=[0,5]

nx=len(dxvec)-1
ny=len(dyvec)-1
nz=len(dzvec)-1
nodelist=[]
llz = 0
for k in range(0, nz+1):
    llz = llz + dzvec[k]
    lly=0
    for j in range(0, ny+1):
        lly = lly + dyvec[j]
        llx = 0
        for i in range(0, nx+1):
            llx = llx + dxvec[i]
            node=NODE()
            node.x=llx
            node.y=lly
            node.z=llz
            nodelist.append(node)

# build connectivity and neighbors
celllist=[]

for k in range(0, nz):
    for j in range(0, ny):
        for i in range(0, nx):
            id = k * nx * ny + j * nx + i
            nc=id
            cell = CELL()
            if i>0:
                cell.neighbors[0] = nc - 1
            if i<nx-1:
                cell.neighbors[1] = nc + 1
            if j>0:
                cell.neighbors[2] = nc - nx
            if j<ny-1:
                cell.neighbors[3] = nc + nx
            if k>0:
                cell.neighbors[4] = nc - nx*ny
            if k<nz-1:
                cell.neighbors[5] = nc + nx * ny
            i0 = k * (nx + 1) * (ny + 1) + j * (nx + 1) + i
            i1 = k * (nx + 1) * (ny + 1) + j * (nx + 1) + i + 1
            i2 = k * (nx + 1) * (ny + 1) + (j + 1) * (nx + 1) + i
            i3 = k * (nx + 1) * (ny + 1) + (j + 1) * (nx + 1) + i + 1
            i4 = (k + 1) * (nx + 1) * (ny + 1) + j * (nx + 1) + i
            i5 = (k + 1) * (nx + 1) * (ny + 1) + j * (nx + 1) + i + 1
            i6 = (k + 1) * (nx + 1) * (ny + 1) + (j + 1) * (nx + 1) + i
            i7 = (k + 1) * (nx + 1) * (ny + 1) + (j + 1) * (nx + 1) + i + 1
            cell.dx = nodelist[i1].x - nodelist[i0].x
            cell.dy = nodelist[i2].y - nodelist[i0].y
            cell.dz = nodelist[i4].z - nodelist[i0].z
            cell.vertices[0] = i0
            cell.vertices[1] = i1
            cell.vertices[2] = i2
            cell.vertices[3] = i3
            cell.vertices[4] = i4
            cell.vertices[5] = i5
            cell.vertices[6] = i6
            cell.vertices[7] = i7
            cell.xc = 0.125 * (nodelist[i0].x+nodelist[i1].x+nodelist[i2].x+nodelist[i3].x+nodelist[i4].x+nodelist[i5].x+nodelist[i6].x+nodelist[i7].x)
            cell.yc = 0.125 * (nodelist[i0].y + nodelist[i1].y + nodelist[i2].y + nodelist[i3].y + nodelist[i4].y + nodelist[i5].y + nodelist[i6].y + nodelist[i7].y)
            cell.zc = 0.125 * (nodelist[i0].z + nodelist[i1].z + nodelist[i2].z + nodelist[i3].z + nodelist[i4].z + nodelist[i5].z + nodelist[i6].z + nodelist[i7].z)
            cell.volume=cell.dx*cell.dy*cell.dz
            celllist.append(cell)

cellvolume=celllist[0].volume
ncell=len(celllist)

print("define properties")
mu_o = 2e-3
ct = 5e-8
poro = 0.1

def single_qwt(permvec):
    for i in range(len(permvec)):
        if permvec[i] >= 20 * 1e-15:
            permvec[i] = 20 * 1e-15
    print("set properties to grid")
    for i in range(0, ncell):
        celllist[i].porosity = poro
        celllist[i].kx = permvec[i]
        celllist[i].ky = permvec[i]
    print("compute transmissibility")
    transvec = []  # for input into network
    for ie in range(0, ncell):
        dx1 = celllist[ie].dx
        dy1 = celllist[ie].dy
        dz1 = celllist[ie].dz
        for j in range(0, 4):
            je = celllist[ie].neighbors[j]
            if je >= 0:
                dx2 = celllist[je].dx
                dy2 = celllist[je].dy
                dz2 = celllist[je].dz
                mt1 = 1.0 / mu_o
                mt2 = 1.0 / mu_o
                if j == 0 or j == 1:
                    mt1 = mt1 * dy1 * dz1
                    mt2 = mt2 * dy2 * dz2
                    k1 = celllist[ie].kx
                    k2 = celllist[je].kx
                    dd1 = dx1 / 2.
                    dd2 = dx2 / 2.
                elif j == 2 or j == 3:
                    mt1 = mt1 * dx1 * dz1
                    mt2 = mt2 * dx2 * dz2
                    k1 = celllist[ie].ky
                    k2 = celllist[je].ky
                    dd1 = dy1 / 2.
                    dd2 = dy2 / 2.
                t1 = mt1 * k1 / dd1
                t2 = mt2 * k2 / dd2
                tt = 1 / (1 / t1 + 1 / t2)
                celllist[ie].trans[j] = tt
                transvec.append(tt)
            else:
                transvec.append(0)

    print("define well condition")
    rw = 0.05
    SS = 3
    length = 3000
    cs = ct * 3.14 * rw * rw * length
    bhp_constant = 20e6
    ddx = dxvec[1] - dxvec[0]
    re = 0.14 * (ddx * ddx + ddx * ddx) ** 0.5
    PI = 2 * 3.14 * ddx * permvec[0] / mu_o / (math.log(re / rw) + SS)
    pwf = bhp_constant  # bottom-hole pressure
    celllist[0].markwell = 1

    print("initial condition")
    p_init = 30.0 * 1e6
    for i in range(0, ncell):
        celllist[i].press = p_init

    print("start transient simulation")
    nt = 4000
    dt = 3000
    press = np.zeros(ncell)
    qwt = np.zeros(nt)  # record flow rate with time
    for t in range(nt):
        print('Time:', t)
        for ie in range(ncell):
            p_i = celllist[ie].press
            rhs = 0
            if celllist[ie].markwell == 1:
                qw = PI * (celllist[ie].press - pwf)
                rhs = -qw
                qwt[t] = qw
            for j in range(4):
                je = celllist[ie].neighbors[j]
                if je >= 0:
                    p_j = celllist[je].press
                    rhs = rhs + celllist[ie].trans[j] * (p_j - p_i)
            dp = rhs * dt / celllist[ie].porosity / ct / celllist[ie].volume
            press[ie] = celllist[ie].press + dp  # for next time step
            if press[ie] < 0:
                print('negative press at ', ie, press[ie])
        for ie in range(ncell):
            celllist[ie].press = press[ie]
    return qwt

