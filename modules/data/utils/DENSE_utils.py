# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 19:18:08 2021

@author: Jerry Xing
"""

import numpy as np
import scipy
import scipy.io as sio
def SVDDenoise(mat, rank=3):
    u, s, vh = np.linalg.svd(mat, full_matrices=False)
    s[rank:] = 0    
    return u@np.diag(s)@vh

def loadStrainMat(filename):
    datamat = sio.loadmat(filename, struct_as_record=False, squeeze_me = True)
    EccDatum, tos = None, None
    
    if 'TransmuralStrainInfo' in datamat.keys():
        EccDatum = SVDDenoise(np.flip(datamat['TransmuralStrainInfo'].Ecc.mid.T, axis=0))
    # if 'strainMatFullResolution' in datamat.keys():
    #     strainMetFullResolution = datamat['strainMatFullResolution']
    # else:
    #     strainMetFullResolution = None
    try:
        strainMetFullResolution = np.flipud(datamat['StrainInfo'].CCmid)
    except:
        strainMetFullResolution = None
        
    
    if 'xs' in datamat.keys():
        tos = datamat['xs'][::-1]
    elif 'TOSAnalysis' in datamat.keys():
        tos = datamat['TOSAnalysis'].TOS[::-1]
    else:
        tos = None
    
    try:
        tos_interp_mid = datamat['TOSAnalysis'].TOSInterploated[datamat['AnalysisInfo'].fv.layerid==3][::-1]
    except:
        tos_interp_mid = None
    # for datum in dataFull:
    #     datum[config['data']['outputType']] = datum['TOSInterploated'][:,datum['AnalysisFv'].layerid==3]

    # if 'TOSInterploated' in datamat.keys():
    #     tos_interp = datamat['TOSInterploated'][::-1]
    # else:
    #     tos_interp = None
    return EccDatum, tos, strainMetFullResolution, tos_interp_mid, datamat

def saveTOS2Mat(tos:np.ndarray, filename:str):
    sio.savemat(filename, {'xs': tos})
# def saveTOS2Mat(data:np.ndarray, filename:str, tos_only = True):
#     if tos_only:
#         pass
#     else:
#         pass
#     sio.savemat(filename, {'xs': tos})

from PyQt5 import QtWidgets
def getScreenSize(displayNr = -1):
    # https://stackoverflow.com/questions/35887237/current-screen-size-in-python3-with-pyqt5
    sizeObject = QtWidgets.QDesktopWidget().screenGeometry(displayNr)
    return sizeObject.height(), sizeObject.width()

def _rect_inter_inner(x1, x2):
    n1 = x1.shape[0]-1
    n2 = x2.shape[0]-1
    X1 = np.c_[x1[:-1], x1[1:]]
    X2 = np.c_[x2[:-1], x2[1:]]
    S1 = np.tile(X1.min(axis=1), (n2, 1)).T
    S2 = np.tile(X2.max(axis=1), (n1, 1))
    S3 = np.tile(X1.max(axis=1), (n2, 1)).T
    S4 = np.tile(X2.min(axis=1), (n1, 1))
    return S1, S2, S3, S4


def _rectangle_intersection_(x1, y1, x2, y2):
    S1, S2, S3, S4 = _rect_inter_inner(x1, x2)
    S5, S6, S7, S8 = _rect_inter_inner(y1, y2)

    C1 = np.less_equal(S1, S2)
    C2 = np.greater_equal(S3, S4)
    C3 = np.less_equal(S5, S6)
    C4 = np.greater_equal(S7, S8)

    ii, jj = np.nonzero(C1 & C2 & C3 & C4)
    return ii, jj


def intersections(x1, y1, x2, y2):
    # https://github.com/sukhbinder/intersection
    """
    INTERSECTIONS Intersections of curves.
       Computes the (x,y) locations where two curves intersect.  The curves
       can be broken with NaNs or have vertical segments.
    usage:
    x,y=intersection(x1,y1,x2,y2)
        Example:
        a, b = 1, 2
        phi = np.linspace(3, 10, 100)
        x1 = a*phi - b*np.sin(phi)
        y1 = a - b*np.cos(phi)
        x2=phi
        y2=np.sin(phi)+2
        x,y,i,j=intersections(x1,y1,x2,y2)
        plt.plot(x1,y1,c='r')
        plt.plot(x2,y2,c='g')
        plt.plot(x,y,'*k')
        plt.show()
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    ii, jj = _rectangle_intersection_(x1, y1, x2, y2)
    n = len(ii)

    dxy1 = np.diff(np.c_[x1, y1], axis=0)
    dxy2 = np.diff(np.c_[x2, y2], axis=0)

    T = np.zeros((4, n))
    AA = np.zeros((4, 4, n))
    AA[0:2, 2, :] = -1
    AA[2:4, 3, :] = -1
    AA[0::2, 0, :] = dxy1[ii, :].T
    AA[1::2, 1, :] = dxy2[jj, :].T

    BB = np.zeros((4, n))
    BB[0, :] = -x1[ii].ravel()
    BB[1, :] = -x2[jj].ravel()
    BB[2, :] = -y1[ii].ravel()
    BB[3, :] = -y2[jj].ravel()

    for i in range(n):
        try:
            T[:, i] = np.linalg.solve(AA[:, :, i], BB[:, i])
        except:
            T[:, i] = np.Inf

    in_range = (T[0, :] >= 0) & (T[1, :] >= 0) & (
        T[0, :] <= 1) & (T[1, :] <= 1)

    xy0 = T[2:, in_range]
    xy0 = xy0.T
    
    iout = ii[in_range] + T[0, in_range].T
    jout = jj[in_range] + T[1, in_range].T
    
    return xy0[:, 0], xy0[:, 1], iout, jout


# https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
def cart2pol(x, y):
    # rho = np.sqrt(x**2 + y**2)
    # phi = np.arctan2(y, x)
    # return(rho, phi)
    # myhypot = @(a,b)sqrt(abs(a).^2+abs(b).^2);
    
    hypot = lambda x,y: np.sqrt(np.abs(x)**2 + np.abs(y)**2)
    th = np.arctan2(y,x);
    r = hypot(x,y);
    
    return th, r
    
# def pol2cart(rho, phi):
def pol2cart(th, r):
    # x = rho * np.cos(phi)
    # y = rho * np.sin(phi)
    # return(x, y)
    x = r*np.cos(th);
    y = r*np.sin(th);
    return x, y 

def spl2patchSA(datamat):
    maxseg = 132

    Ccell   = datamat['ROIInfo'].RestingContour
    origin  = datamat['AnalysisInfo'].PositionA
    posB    = datamat['AnalysisInfo'].PositionB
    flag_clockwise = datamat['AnalysisInfo'].Clockwise
    Nseg    = 18

    # total number of theta samples per segment
    Nperseg = int(np.floor(maxseg/Nseg))
    N = int(Nperseg*Nseg)

    # full enclosing contour
    C = Ccell.copy()
    for cidx in range(len(C)):
        C[cidx] = np.concatenate([C[cidx], np.nan*np.ones((1,2))])
    C = np.concatenate([c for c in C])

    # initial angle
    # atan2 -> arctan2
    theta0 = np.arctan2(posB[1]-origin[1],posB[0]-origin[0])

    # angular range
    if flag_clockwise:
        theta = np.linspace(0,2*np.pi,N+1).reshape([1,-1])
    else:
        theta = np.linspace(2*np.pi,0,N+1).reshape([1,-1])

    theta = theta[:,:-1] + theta0


    # radial range
    tmp,r = cart2pol(C[:,0]-origin[0],C[:,1]-origin[1])
    mxrad = np.ceil(max(r))
    rad = np.array([0, 2*mxrad])

    # spokes
    THETA,RAD = np.meshgrid(theta,rad)
    THETA,RAD = THETA.T,RAD.T
    X,Y = pol2cart(THETA,RAD)

    xspoke = X.T+origin[0]
    xspoke = np.concatenate([xspoke, np.nan*np.ones((1, xspoke.shape[1]))])

    yspoke = Y.T+origin[1]
    yspoke = np.concatenate([yspoke, np.nan*np.ones((1, xspoke.shape[1]))])

    # find intersections
    x,y,_,_ = intersections(xspoke.flatten(order='F'),
                              yspoke.flatten(order='F'),
                              Ccell[0][:,0], Ccell[0][:,1])

    # record points
    eppts = np.concatenate((x[:,None], y[:,None]), axis=1)


    # find intersections
    x,y,_,_ = intersections(xspoke.flatten(order='F'),
                              yspoke.flatten(order='F'),
                              Ccell[1][:,0], Ccell[1][:,1])


    # record points
    enpts = np.concatenate((x[:,None], y[:,None]), axis=1)

    # number of lines
    Nline = 6

    # vertices
    X = np.nan*np.ones((N, Nline))
    Y = np.nan*np.ones((N, Nline))

    w = np.linspace(0,1,Nline)
    # for k = 1:Nline
    for k in range(Nline):
        X[:,k] = w[k]*enpts[:,0] + (1-w[k])*eppts[:,0]
        Y[:,k] = w[k]*enpts[:,1] + (1-w[k])*eppts[:,1]
    v = np.concatenate((X.flatten(order='F')[:, None], Y.flatten(order='F')[:,None]), axis=1)

    # 4-point faces
    f = np.zeros(((Nline-1)*N,4)).astype(int)
    tmp1 = np.arange(N)[:, None]
    tmp2 = np.append(np.arange(1,N), 0)[:, None]
    tmp = np.hstack((tmp1, tmp2))
    for k in range(Nline-1):
        rows = k*N + np.arange(N)
        f[rows,:] = np.hstack((tmp, np.fliplr(tmp)+N)) + k*N
    Nface = f.shape[0]


    # ids
    ids = np.repeat(np.arange(Nseg),Nperseg,0) + 1 # +1 to match the index format of MATLAB
    ids = np.repeat(ids[:, None], Nline - 1, 1)
    sectorid = ids.flatten(order='F')

    layerid = np.repeat(np.arange(Nline-1), N) + 1



    # face locations (average of vertices)
    # pface = NaN(Nface,2);
    pface = np.nan*np.ones((Nface,2))
    for k in [0, 1]:
        vk = v[:,k]
        pface[:,k] = np.mean(vk[f],1)

    # orientation (pointed towards center)
    ori,rad = cart2pol(origin[0]-pface[:,0], origin[1]-pface[:,1])


    # gather output data
    fv = {'vertices':    v,
          'faces':       f + 1,    # +1 to match the index format of MATLAB
          'sectorid':    sectorid,
          'layerid':     layerid,
          'orientation': ori
        }
    return fv

def rectfv2rectfv(fv1, vals1, fv2):
    Nfaces1 = fv1['faces'].shape[0]
    Nfaces2 = fv2['faces'].shape[0]
    centers1 = np.zeros((Nfaces1, 2))
    centers2 = np.zeros((Nfaces2, 2))
    for faceIdx in range(Nfaces1):
        centers1[faceIdx,:] = np.mean(fv1['vertices'][fv1['faces'][faceIdx,:]-1,:], axis=0)
    for faceIdx in range(Nfaces2):
        centers2[faceIdx,:] = np.mean(fv2['vertices'][fv2['faces'][faceIdx,:]-1,:], axis=0)
    
    # centers2GridX, centers2GridY = np.meshgrid(centers2[:,0],centers2[:,1])
    # mask = (xi > 0.5) & (xi < 0.6) & (yi > 0.5) & (yi < 0.6)
    # vals2 = griddata((centers1[:,0],centers1[:,1]),vals1,(centers2GridX,centers2GridY),method='nearest')
    vals2 = scipy.interpolate.griddata(centers1,vals1,centers2,method='linear')
    
    # interp = scipy.interpolate.LinearNDInterpolator(centers1, vals1)
    return vals2

def getStrainMatFull(datamat, fv = None):
    if fv is None:
        fv = spl2patchSA(datamat)
    NFrames = datamat['ImageInfo'].Xunwrap.shape[-1]
    NFacesPerLayer = np.sum(fv['layerid'] == 1)
    strainMatFull = np.zeros((NFacesPerLayer, NFrames))
    for frameIdx in range(NFrames):
        CCinNewFv = rectfv2rectfv({'faces': datamat['StrainInfo'].Faces, 'vertices': datamat['StrainInfo'].Vertices}, datamat['StrainInfo'].CC[:,frameIdx], fv)
        strainMatFull[:,frameIdx] = CCinNewFv[fv['layerid']==3]
    return strainMatFull