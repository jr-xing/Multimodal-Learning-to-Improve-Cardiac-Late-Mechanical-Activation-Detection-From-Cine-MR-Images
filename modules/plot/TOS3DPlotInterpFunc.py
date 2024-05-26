# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 09:24:30 2020

@author: remus
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from matplotlib.collections import LineCollection

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
# register Axes3D class with matplotlib by importing Axes3D
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D
from matplotlib import cm

def text3d(ax, xyz, s, zdir="z", size=None, angle=0, usetex=False, **kwargs):
    '''
    http://members.cbio.mines-paristech.fr/~nvaroquaux/tmp/matplotlib/examples/mplot3d/pathpatch3d_demo.html
    Plots the string 's' on the axes 'ax', with position 'xyz', size 'size',
    and rotation angle 'angle'.  'zdir' gives the axis which is to be treated
    as the third dimension.  usetex is a boolean indicating whether the string
    should be interpreted as latex or not.  Any additional keyword arguments
    are passed on to transform_path.

    Note: zdir affects the interpretation of xyz.
    '''
    x, y, z = xyz
    if zdir == "y":
        xy1, z1 = (x, z), y
    elif zdir == "y":
        xy1, z1 = (y, z), x
    else:
        xy1, z1 = (x, y), z

    text_path = TextPath((0, 0), s, size=size, usetex=usetex)
    trans = Affine2D().rotate(angle).translate(xy1[0], xy1[1])

    p1 = PathPatch(trans.transform_path(text_path), **kwargs)
    ax.add_patch(p1)
    art3d.pathpatch_2d_to_3d(p1, z=z1, zdir=zdir)

# def get_DENSE_data_from_slice_mat(mat, data_type):
#     if data_type == 'vertices':
#         pass

def TOS3DPlotInterp_OLD(dataOfPatient, tos_key = 'TOSInterploated', 
                    spatial_location_key = 'SequenceInfo',
                    title = None, alignCenters = True, restoreOriSlices = False, vmax = None, axe = None):  
    # interpolate = False
    interpolate = True
    # restoreOriSlices = True
    points_interp1d_method = 'quadratic'
    # tos_interp1d_method = 'linear'
    tos_interp1d_method = 'nearest'
    # 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next'
    ifMergeDataByPatient = False
    
    # Align by image center
    # patientVertices = [sliceData['AnalysisFv'].vertices for sliceData in dataOfPatient]
    patientVertices = [sliceData['AnalysisFv']['vertices'] for sliceData in dataOfPatient]
    patientVerticesXMean = np.mean(np.concatenate([vertices[:,0] for vertices in patientVertices]))
    patientVerticesYMean = np.mean(np.concatenate([vertices[:,1] for vertices in patientVertices]))    

    NSlicesInterp = 50
    # midLayerLen = sum(dataOfPatient[0]['AnalysisFv'].layerid ==3)
    midLayerLen = sum(dataOfPatient[0]['AnalysisFv']['layerid'] ==3)
    # xsMatInterp, ysMatInterp, TOSMatInterp = [np.zeros((NSlicesInterp, midLayerLen))] * 4
    xsMatInterp = np.zeros((NSlicesInterp, midLayerLen))
    ysMatInterp = np.zeros((NSlicesInterp, midLayerLen))
    TOSMatInterp = np.zeros((NSlicesInterp, midLayerLen))
    
    # Sort slice by spatial location
    sliceSpatialLocOrder = np.argsort([datum[spatial_location_key] for datum in dataOfPatient])
    dataOfPatient = [dataOfPatient[idx] for idx in sliceSpatialLocOrder]
    
    NSlicesOri = len(dataOfPatient)
    # NSectors = len(set(dataOfPatient[0]['AnalysisFv'].sectorid))
    NSectors = len(set(dataOfPatient[0]['AnalysisFv']['sectorid']))
    sectors = []
    # for sectorIdx in range(NSectors): sectors.append({})
    # Update Each Sector's Information using the first slice
    slice0Data = dataOfPatient[0]
    for sectorIdx, sectorId in enumerate(range(1, NSectors+1)):
        # sectorInSliceFaces = slice0Data['AnalysisFv'].faces[slice0Data['AnalysisFv'].sectorid==sectorId]
        sectorInSliceFaces = slice0Data['AnalysisFv']['faces'][slice0Data['AnalysisFv']['sectorid']==sectorId]
        sectorInSliceVertices = slice0Data['AnalysisFv']['vertices'][sectorInSliceFaces-1]
        sectors.append({'leftMostX': sectorInSliceVertices[0,0,0], 'leftMostY':sectorInSliceVertices[0,0,1]})
    
    xsMatOri = np.zeros((NSlicesOri, midLayerLen))
    ysMatOri = np.zeros((NSlicesOri, midLayerLen))
    TOSMatOri = np.zeros((NSlicesOri, midLayerLen))
    
    for sliceIdx, sliceData in enumerate(dataOfPatient):            
        # sliceMidLayerFaces = [sliceData['AnalysisFv'].faces[idx] for idx in range(len(sliceData['AnalysisFv'].layerid)) if sliceData['AnalysisFv'].layerid[idx]==3]
        # sliceMidLayerVertices = np.concatenate([np.expand_dims(sliceData['AnalysisFv'].vertices[sliceMidLayerFace-1], axis=0) for sliceMidLayerFace in sliceMidLayerFaces])
        sliceMidLayerFaces = [sliceData['AnalysisFv']['faces'][idx] for idx in range(len(sliceData['AnalysisFv']['layerid'])) if sliceData['AnalysisFv']['layerid'][idx]==3]
        sliceMidLayerVertices = np.concatenate([np.expand_dims(sliceData['AnalysisFv']['vertices'][sliceMidLayerFace-1], axis=0) for sliceMidLayerFace in sliceMidLayerFaces])
        sliceMidLayerXs = sliceMidLayerVertices[:,:,0]
        sliceMidLayerYs = sliceMidLayerVertices[:,:,1]
        xsMatOri[sliceIdx, :] = np.mean(sliceMidLayerXs, axis=1)
        ysMatOri[sliceIdx, :] = np.mean(sliceMidLayerYs, axis=1)
        if tos_key in sliceData.keys():
            # TOSMatOri[sliceIdx, :] = sliceData[tos_key]#[0, sliceData['AnalysisFv'].layerid == 3]
            TOSMatOri[sliceIdx, :] = sliceData[tos_key][:,:126]
            hasTOS = True
        else:
            hasTOS = False
            TOSMatOri[sliceIdx, :] = 0
            
    if alignCenters:
        for sliceIdx in range(len(dataOfPatient)):
            sliceXsMean = np.mean(xsMatOri[sliceIdx,:])
            sliceYsMean = np.mean(ysMatOri[sliceIdx,:])
            xsMatOri[sliceIdx,:] = xsMatOri[sliceIdx,:] - sliceXsMean + patientVerticesXMean
            ysMatOri[sliceIdx,:] = ysMatOri[sliceIdx,:] - sliceYsMean + patientVerticesYMean

    sliceSpatialLocsOri = np.array([sliceData[spatial_location_key] for sliceData in dataOfPatient])
    sliceSpatialLocsMin, sliceSpatialLocsMax = np.min(sliceSpatialLocsOri), np.max(sliceSpatialLocsOri)
    sliceSpatialLocsInterp = np.linspace(sliceSpatialLocsMin, sliceSpatialLocsMax, NSlicesInterp)
    # Restore original locations
    if restoreOriSlices:
        for sliceIdx, sliceLoc in enumerate(sliceSpatialLocsOri):
            closestIdx = np.argmin(np.abs(sliceSpatialLocsInterp - sliceLoc))
            sliceSpatialLocsInterp[closestIdx] = sliceLoc
    
    
    zsMatOri = np.repeat(sliceSpatialLocsOri.reshape(-1, 1), axis=1, repeats=midLayerLen)
    zsMatInterp = np.repeat(sliceSpatialLocsInterp.reshape(-1, 1), axis=1, repeats=midLayerLen)
    # Interploation
    for ringLoc in range(xsMatOri.shape[1]):
        # For each location on the ring, i.e. column of matOri            
        xsColOri = xsMatOri[:, ringLoc]
        ysColOri = ysMatOri[:, ringLoc]            
        TOSColOri = TOSMatOri[:, ringLoc]
        
        interpFuncX = interp1d(sliceSpatialLocsOri, xsColOri, kind=points_interp1d_method)
        interpFuncY = interp1d(sliceSpatialLocsOri, ysColOri, kind=points_interp1d_method)
        interpFuncTOS = interp1d(sliceSpatialLocsOri, TOSColOri, kind=tos_interp1d_method)
        xsMatInterp[:, ringLoc] = interpFuncX(sliceSpatialLocsInterp)
        ysMatInterp[:, ringLoc] = interpFuncY(sliceSpatialLocsInterp)
        TOSMatInterp[:, ringLoc] = interpFuncTOS(sliceSpatialLocsInterp)
            
    
    xsFlat, ysFlat, zsFlat, TOSFlat = [data.flatten() for data in [xsMatInterp,ysMatInterp,zsMatInterp, TOSMatInterp]]
    xsOriFlat, ysOriFlat, zsOriFlat, TOSOriFlat = [data.flatten() for data in [xsMatOri,ysMatOri,zsMatOri, TOSMatOri]]
    
    xsOrder = np.argsort(xsOriFlat)
    xsFlatOrdered = xsOriFlat[xsOrder]
    ysFlatOrdered = ysOriFlat[xsOrder]
    zsFlatOrdered = zsOriFlat[xsOrder]
    
    xsGrid, ysGrid = np.meshgrid(xsFlatOrdered, ysFlatOrdered)
    # zsGrid, _ = np.meshgrid(zsFlatOrdered,zsFlatOrdered)                    
    # zsGrid = np.ones((len(xsFlatOrdered), len(ysFlatOrdered))) * np.nan
    # for point_idx in range(len(xsFlatOrdered)):
    #     zsGrid[]
    
    # xsFlat, ysFlat, zsFlat, TOSFlat = [data.flatten() for data in [xsMatOri,ysMatOri,zsMatOri, TOSMatOri]]
    if axe is None:
        fig = plt.figure()
        axe = fig.gca(projection='3d')
    # axe.plot_surface(xsGrid[::10,::10], ysGrid[::10,::10], (xsGrid[::10,::10]-65)**2+(ysGrid[::10,::10]-65)**2)
    # axe.plot_surface(xsGrid, ysGrid, (xsGrid-65)**2+(ysGrid-65)**2)
    # axe.plot_surface(xsGrid, ysGrid, zsGrid)
    # axe.plot_trisurf(xsFlatOrdered, ysFlatOrdered, zsFlatOrdered)   
    # if hasTOS:
    #     color = TOSFlat
    # else:
    #     color = zsFlat
    
    if interpolate:
        color = TOSFlat if hasTOS else zsFlat
        scatterPlot = axe.scatter(xsFlat, ysFlat, zsFlat, c = color, cmap='jet', zorder = 2, vmax = vmax, vmin = 17)
        # scatterPlot = axe.plot_surface(xsGrid, ysGrid, zsGrid, cmap='jet', vmax = vmax, vmin = 17)
        # ax.contour3D(X, Y, Z, 50, cmap='binary')
    else:
        color = TOSOriFlat if hasTOS else zsOriFlat
        scatterPlot = axe.scatter(xsOriFlat, ysOriFlat, zsOriFlat, c = color, cmap='jet', zorder = 2, vmax = vmax, vmin = 17)
    axe.view_init(elev=30., azim=-10)
    # axe.view_init(elev=90., azim=-10)
    axe.set_xlabel('X')
    axe.set_ylabel('Y')
    axe.set_zlabel('Spatial Location')
    axe.set_axis_off()
    # axe.set_title('TOS Surface' + '\n ' + patientID2Show.replace('//', '-') + ('\n (FAKE TOS)' if not hasTOS else ''))
    if title is not None:
        axe.set_title(title)
    # axe.set_zlim(np.min(zsFlat) - 15, np.max(zsFlat)+15)
    # plt.colorbar(scatterPlot, ax = axe)
    
    
    # Try Draw surface
    
    
    # Draw Sectors
    # https://matplotlib.org/3.1.0/gallery/mplot3d/text3d.html
    # https://github.com/pyplot-examples/pyplot-3d-wedge/blob/master/wedge.py
    draw_sectors = False
    if draw_sectors:
        centerX = np.mean(xsFlat)
        centerY = np.mean(ysFlat)
        centerZ = np.min(zsFlat) - 10
        patches = []
        lines = []
        for sectorIdx, sector in enumerate(sectors):
            leftMostX = sectors[sectorIdx]['leftMostX']# 
            leftMostY = sectors[sectorIdx]['leftMostY']# - centerY
            rightMostX = sectors[(sectorIdx+1)%NSectors]['leftMostX']# - centerX
            rightMostY = sectors[(sectorIdx+1)%NSectors]['leftMostY']# - centerY
            startAngle = np.arctan((leftMostY - centerY) / (leftMostX - centerX)) * 180 / np.pi
            endAngle = np.arctan((rightMostY- centerY) / (rightMostX- centerX)) * 180 / np.pi
            
            lines.append([(leftMostX,leftMostY),(centerX,centerY)])
            
            # axe.text(leftMostX, leftMostY, centerZ, "red", color='red')
            # sectorNames = ['LAD']*3*2 + ['RCA']*3*2 + ['LCX']*3*2
            # sectorNames = [f'LAD{idx}' for idx in range(1,7)] + [f'RCA{idx}' for idx in range(1,7)] + [f'LCX{idx}' for idx in range(1,7)]
            # sectorNames = sectorNames[::-1]
            text3d(axe, ((leftMostX + rightMostX)/2, (leftMostY+rightMostY)/2, centerZ), f'S{sectorIdx+1}', zdir = 'z', size = 1)
            # text3d(axe, ((leftMostX + rightMostX)/2-1, (leftMostY+rightMostY)/2, centerZ), sectorNames[sectorIdx], zdir = 'z', size = 1)
            
            # wedge = Wedge((centerX, centerY), 10, startAngle, endAngle, color='green', alpha=0.4)
            # axe.add_patch(wedge)
            # art3d.pathpatch_2d_to_3d(wedge, z=centerZ, zdir='z')
            # patches.append(wedge)
        linesC = LineCollection(lines,zorder=1,color='green',lw=3, alpha = 0.4)
        axe.add_collection3d(linesC,zs=centerZ)
        # p = art3d.Patch3DCollection(patches, alpha=0.4, zorder = 1)
        # axe.add_collection3d(p)
    # eye = plt.imread('./eye_plot.gif')
    # eye = (eye - np.min(eye)) / (np.max(eye) - np.min(eye))
    # eyeH, eyeW = eye.shape[:2]
    # fakeImgXs = np.arange(-eyeW/2 + centerX,eyeW/2 + centerX)
    # fakeImgYs = np.arange(-eyeH/2 + centerY,eyeH/2 + centerY)
    # fakeImgXsGrid, fakeImgYsGrid = np.meshgrid(fakeImgXs, fakeImgYs)
    
    
# def sort_slice_by_spatial_location(data):
#     pass

def generate_3D_Activation_map(data: list, 
                               tos_key = 'TOS126', 
                               spatial_location_key = 'slice_spatial_location',
                               interpolate = True,
                               spatial_interp_method = 'cubic',
                               value_interp_method = 'linear',
                               title = None, 
                               slice_spatial_order = 'increasing',
                               align_centers = True, 
                               keep_ori_slice_locations = True, 
                               vmax = None, vmin = 17, axe = None,
                               view_elev=30, view_azim=-10, view_roll=0,
                               colorbar=False, axis_off=False, TOS_scaling_factor=1, TOS_offset=0):
    
    # 1) Sort slice by spatial location
    if slice_spatial_order == 'increasing':
        sliceSpatialLocOrder = np.argsort([datum[spatial_location_key] for datum in data])
    elif slice_spatial_order == 'decreasing':
        sliceSpatialLocOrder = np.argsort([datum[spatial_location_key] for datum in data])[::-1]
    data = [data[idx] for idx in sliceSpatialLocOrder]

    # 2) Align by centering
    # patientVertices = [sliceData['AnalysisFv'].vertices for sliceData in dataOfPatient]
    if align_centers:        
        # print('ALIGN')
        vertices_of_each_slice = [sliceData['AnalysisFv']['vertices'] for sliceData in data]
        vertices_x_mean_of_all_slices = np.mean(np.concatenate([slice_vertices[:,0] for slice_vertices in vertices_of_each_slice]))
        vertices_y_mean_of_all_slices = np.mean(np.concatenate([slice_vertices[:,1] for slice_vertices in vertices_of_each_slice]))
        for slice_idx, slice_data in enumerate(data):
            slice_data['AnalysisFv']['vertices'][:,0] -= vertices_x_mean_of_all_slices
            slice_data['AnalysisFv']['vertices'][:,1] -= vertices_y_mean_of_all_slices
    
    # 3) Build xs,ys, zs in grid
    n_sectors = 126
    n_slices = len(data)
    xs_raw_grid = np.zeros((n_slices, n_sectors))
    ys_raw_grid = np.zeros((n_slices, n_sectors))
    zs_raw_grid = np.zeros((n_slices, n_sectors))
    TOS_raw_grid = np.zeros((n_slices, n_sectors))
    zs_denom = 8
    for slice_idx, slice_data in enumerate(data):
        slice_mid_layer_faces = [slice_data['AnalysisFv']['faces'][face_idx] for face_idx in range(len(slice_data['AnalysisFv']['layerid'])) if slice_data['AnalysisFv']['layerid'][face_idx]==3]
        slice_mid_layer_vertices = np.concatenate([np.expand_dims(slice_data['AnalysisFv']['vertices'][slice_mid_layer_face-1], axis=0) for slice_mid_layer_face in slice_mid_layer_faces])
        slice_mid_layer_xs = slice_mid_layer_vertices[:,:,0]
        slice_mid_layer_ys = slice_mid_layer_vertices[:,:,1]
        xs_raw_grid[slice_idx, :] = np.mean(slice_mid_layer_xs, axis=1)
        ys_raw_grid[slice_idx, :] = np.mean(slice_mid_layer_ys, axis=1)
        zs_raw_grid[slice_idx, :] = slice_data[spatial_location_key] / zs_denom
        TOS_raw_grid[slice_idx, :] = np.squeeze(slice_data[tos_key])[:n_sectors] * TOS_scaling_factor + TOS_offset
        
    xs_raw_flat, ys_raw_flat, zs_raw_flat, TOS_raw_flat = [grid_data.flatten() for grid_data in [xs_raw_grid, ys_raw_grid, zs_raw_grid, TOS_raw_grid]]
    # 3) Interpolate
    if interpolate:
        from scipy import interpolate as sci_interpolate
        
        # from scipy.interpolate import interpn
        # aa = interpn((xs_flat_raw, ys_flat_raw, zs_flat_raw), TOS_flat_raw, (50,50,100))
        
        sector_indices_raw_flat = np.arange(n_sectors)
        zs_raw_flat = np.array([slice_data[spatial_location_key] for slice_data in data])    
        sector_indices_raw_grid, zs_raw_grid = np.meshgrid(sector_indices_raw_flat, zs_raw_flat)
        # sptial_interp_method = 'cubic'
        # TOS_interp_method = 'linear'
        # TOS_interp_method = 'nearest'
        # print(sector_indices_raw_flat.shape)
        # print(zs_raw_flat.shape)
        # print(xs_raw_grid.shape)
        xs_interp_func = sci_interpolate.interp2d(sector_indices_raw_flat, zs_raw_flat, xs_raw_grid, kind=spatial_interp_method)
        
        # xx = sci_interpolate.interpn((sector_indices_raw_flat, zs_raw_flat[::-1]), xs_raw_grid.T, (sector_indices_interp_flat, zs_interp_flat[::-1]), method='linear')
        
        ys_interp_func = sci_interpolate.interp2d(sector_indices_raw_flat, zs_raw_flat, ys_raw_grid, kind=spatial_interp_method)        
        
        n_sectors_interp = n_sectors
        n_slices_interp = 50
        sector_indices_interp_flat = np.linspace(0, n_sectors-1, n_sectors_interp)
        if slice_spatial_order == 'increasing':
            zs_interp_flat = np.linspace(np.min(zs_raw_flat), np.max(zs_raw_flat), n_slices_interp)
        elif slice_spatial_order == 'decreasing':
            zs_interp_flat = np.linspace(np.min(zs_raw_flat), np.max(zs_raw_flat), n_slices_interp)[::-1]
        
        sector_indices_interp_grid, zs_interp_grid = np.meshgrid(sector_indices_interp_flat, zs_interp_flat)
        xs_interp_grid = xs_interp_func(sector_indices_interp_flat, zs_interp_flat)
        ys_interp_grid = ys_interp_func(sector_indices_interp_flat, zs_interp_flat)
                
        if value_interp_method in ['linear', 'cubic', 'quintic']:
            TOS_interp_func = sci_interpolate.interp2d(sector_indices_raw_flat, zs_raw_flat, TOS_raw_grid, kind=value_interp_method)
            TOS_interp_grid = TOS_interp_func(sector_indices_interp_flat, zs_interp_flat)
        elif value_interp_method in ['nearest']:
            from scipy.interpolate import NearestNDInterpolator
            # print(sector_indices_raw_flat.shape)
            # print(zs_raw_flat.shape)
            # print(len(list(zip(sector_indices_raw_flat, zs_raw_flat))))
            # print(TOS_raw_flat.shape)
            TOS_interp_func = NearestNDInterpolator(list(zip(sector_indices_raw_grid.flatten(), zs_raw_grid.flatten())), TOS_raw_flat)
            TOS_interp_grid = TOS_interp_func(sector_indices_interp_grid, zs_interp_grid)
        # elif value_interp_method in ['splinef2d']:
        #     from scipy.interpolate import interpn
        #     TOS_interp_func = interpn(list(zip(sector_indices_raw_grid.flatten(), zs_raw_grid.flatten())), TOS_raw_flat)
        #     TOS_interp_grid = TOS_interp_func(sector_indices_interp_grid, zs_interp_grid)
        
        # TOS_interp_grid = TOS_interp_func(sector_indices_interp_flat, zs_interp_flat).T
        # TOS_interp_grid = TOS_interp_func(sector_indices_interp_flat, zs_interp_flat)
        # print(TOS_interp_grid)
        
        xs_grid = xs_interp_grid
        ys_grid = ys_interp_grid
        zs_grid = zs_interp_grid
        TOS_grid = TOS_interp_grid
    else:
        xs_grid = xs_raw_grid
        ys_grid = ys_raw_grid
        zs_grid = zs_raw_grid
        TOS_grid = TOS_raw_grid
        
    smooth_interpolated_grid = True
    if smooth_interpolated_grid:
        smooth_interpolated_level = 1
        def SVDDenoise(mat, rank=3):
            u, s, vh = np.linalg.svd(mat, full_matrices=False)
            s[rank:] = 0    
            return u@np.diag(s)@vh
        xs_grid = SVDDenoise(xs_grid, smooth_interpolated_level)
        ys_grid = SVDDenoise(ys_grid, smooth_interpolated_level)
        
    # 4) Plot
    xs_flat, ys_flat, zs_flat, TOS_flat = [grid_data.flatten() for grid_data in [xs_grid, ys_grid, zs_grid, TOS_grid]]
    fig = plt.figure()
    # axe = fig.gca(projection='3d')
    axe = fig.add_subplot(projection = '3d')

    # print(np.max(TOS_flat))
    scatter_plot = axe.scatter(xs_flat, ys_flat, zs_flat, s = 50, c = TOS_flat, cmap='jet', zorder = 2, vmax = vmax, vmin = vmin)
    axe.view_init(elev=view_elev, azim=view_azim, roll=view_roll)
    axe.set_xlabel('X')
    axe.set_ylabel('Y')
    axe.set_zlabel('Spatial Location')
    if axis_off:
        axe.set_axis_off()
    # axe.set_title('TOS Surface' + '\n ' + patientID2Show.replace('//', '-') + ('\n (FAKE TOS)' if not hasTOS else ''))
    if title is not None:
        axe.set_title(title)
    if colorbar:
        plt.colorbar(scatter_plot, ax = axe)

    

    
    
    # view_names = ['front-left-side', 'front-right-side', 'back-side']
    # for view_name in view_names:
    #     if view_name == 'front-left-side':
    #         elev, azim, roll = 55, 75, 90
    #     elif view_name == 'front-right-side':
    #         elev, azim, roll = -25, 80, 90
    #     elif view_name == 'back-side':
    #         elev, azim, roll = -110, 50, 90
    #     else:
    #         raise ValueError('view_name should be one of the following: front-left-side, front-right-side, back-side')
    # axe.view_init(elev=elev, azim=azim, roll=roll) 
    
    draw_sectors = False
    if draw_sectors:
        sector_lines_center_x = np.mean(xs_flat)
        sector_lines_center_y = np.mean(ys_flat)
        sector_lines_center_z = np.min(zs_flat) - 10
        patches = []
        lines = []
        sector_names = ['inferoseptal', 'inferior', 'inferolateral', 'anterolateral', 'anterior', 'anteroseptal']
        # sectors = data[0]
        for sector_idx in range(6):
            start_sub_sector_idx = (n_sectors // 6) * sector_idx
            start_sub_sector_vertices = data[0]['AnalysisFv']['vertices'][data[0]['AnalysisFv']['faces'][start_sub_sector_idx]-1] # should be a (4,2) array
            start_sub_sector_1st_x, start_sub_sector_1st_y = start_sub_sector_vertices[0,:]
            # sector_center_x, sector_center_y = np.mean(sector_vertices[2:4,:], axis=0)
            
            # sector_2nd_x, sector_2nd_y = sector_vertices[1,:]
            
            # sector_end_angle = np.arctan((sector_1st_y - sector_lines_center_y) / (sector_1st_x - sector_lines_center_x)) * 180 / np.pi
            
            lines.append([(start_sub_sector_1st_x,start_sub_sector_1st_y),
                          (sector_lines_center_x,sector_lines_center_y)])
            
            mid_sub_sector_idx = (n_sectors // 6) * sector_idx + n_sectors // 12
            mid_sub_sector_vertices = data[0]['AnalysisFv']['vertices'][data[0]['AnalysisFv']['faces'][mid_sub_sector_idx]-1] # should be a (4,2) array
            mid_sub_sector_1st_x, mid_sub_sector_1st_y = mid_sub_sector_vertices[0,:]
            mid_sub_sector_center_x, mid_sub_sector_center_y = np.mean(mid_sub_sector_vertices, axis=0)
            mid_sub_sector_angle = np.arctan((mid_sub_sector_1st_y - sector_lines_center_y) / (mid_sub_sector_1st_x - sector_lines_center_x)) * 180 / np.pi
            text3d(axe, (mid_sub_sector_center_x, mid_sub_sector_center_y, sector_lines_center_z), sector_names[sector_idx], 
                   # angle = mid_sub_sector_angle, 
                   angle=view_azim+45,
                   zdir = 'z', size = 2)
            
    #         leftMostX = sectors[sectorIdx]['leftMostX']# 
    #         leftMostY = sectors[sectorIdx]['leftMostY']# - centerY
    #         rightMostX = sectors[(sectorIdx+1)%NSectors]['leftMostX']# - centerX
    #         rightMostY = sectors[(sectorIdx+1)%NSectors]['leftMostY']# - centerY
    #         startAngle = np.arctan((leftMostY - centerY) / (leftMostX - centerX)) * 180 / np.pi
    #         endAngle = np.arctan((rightMostY- centerY) / (rightMostX- centerX)) * 180 / np.pi
            
    #         lines.append([(leftMostX,leftMostY),(centerX,centerY)])
            
    #         # axe.text(leftMostX, leftMostY, centerZ, "red", color='red')
    #         # sectorNames = ['LAD']*3*2 + ['RCA']*3*2 + ['LCX']*3*2
    #         # sectorNames = [f'LAD{idx}' for idx in range(1,7)] + [f'RCA{idx}' for idx in range(1,7)] + [f'LCX{idx}' for idx in range(1,7)]
    #         # sectorNames = sectorNames[::-1]
    #         text3d(axe, ((leftMostX + rightMostX)/2, (leftMostY+rightMostY)/2, centerZ), f'S{sectorIdx+1}', zdir = 'z', size = 1)
    #         # text3d(axe, ((leftMostX + rightMostX)/2-1, (leftMostY+rightMostY)/2, centerZ), sectorNames[sectorIdx], zdir = 'z', size = 1)
            
    #         # wedge = Wedge((centerX, centerY), 10, startAngle, endAngle, color='green', alpha=0.4)
    #         # axe.add_patch(wedge)
    #         # art3d.pathpatch_2d_to_3d(wedge, z=centerZ, zdir='z')
    #         # patches.append(wedge)
        linesC = LineCollection(lines,zorder=1,color='green',lw=3, alpha = 0.4)
        axe.add_collection3d(linesC,zs=sector_lines_center_z)
    return fig, axe