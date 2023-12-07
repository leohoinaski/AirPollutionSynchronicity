#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 17:49:55 2022

@author: leohoinaski
"""
# import os
import numpy as np
from datetime import datetime
import pandas as pd
# import netCDF4 as nc
# from numpy.lib.stride_tricks import sliding_window_view
import pyproj
from shapely.geometry import Point
import geopandas as gpd
from ismember import ismember
# import wrf


def dailyAverage(datesTime, data):
    if len(data.shape) > 3:
        daily = datesTime.groupby(['year', 'month', 'day']).count()
        dailyData = np.empty(
            (daily.shape[0], data.shape[1], data.shape[2], data.shape[3]))
        for day in range(0, daily.shape[0]):
            findArr = (datesTime['year'] == daily.index[day][0]) & \
                (datesTime['month'] == daily.index[day][1]) & \
                (datesTime['day'] == daily.index[day][2])
            dailyData[day, :, :, :] = data[findArr, :, :, :].mean(axis=0)
    else:
        daily = datesTime.groupby(['year', 'month', 'day']).count()
        dailyData = np.empty((daily.shape[0], data.shape[1], data.shape[2]))
        for day in range(0, daily.shape[0]):
            findArr = (datesTime['year'] == daily.index[day][0]) & \
                (datesTime['month'] == daily.index[day][1]) & \
                (datesTime['day'] == daily.index[day][2])
            dailyData[day, :, :] = data[findArr, :, :].mean(axis=0)
    daily = daily.reset_index()
    return dailyData, daily


def dailyRainWRF(datesTime, data):
    if len(data.shape) > 3:
        daily = datesTime.groupby(['year', 'month', 'day']).count()
        dailyData = np.empty(
            (daily.shape[0], data.shape[1], data.shape[2], data.shape[3]))
        for day in range(0, daily.shape[0]):
            findArr = (datesTime['year'] == daily.index[day][0]) & \
                (datesTime['month'] == daily.index[day][1]) & \
                (datesTime['day'] == daily.index[day][2])
            # findArr=findArr.reset_index()
            findArr[np.where(findArr)[0][:-1]] = False
            dailyData[day, :, :, :] = data[findArr, :, :, :]
    else:
        daily = datesTime.groupby(['year', 'month', 'day']).count()
        dailyData = np.empty((daily.shape[0], data.shape[1], data.shape[2]))
        for day in range(0, daily.shape[0]):
            findArr = (datesTime['year'] == daily.index[day][0]) & \
                (datesTime['month'] == daily.index[day][1]) & \
                (datesTime['day'] == daily.index[day][2])
            findArr[np.where(findArr)[0][:-1]] = False
            dailyData[day, :, :] = data[findArr, :, :]
    daily = daily.reset_index()
    return dailyData, daily


def monthlyAverage(datesTime, data):
    monthly = datesTime.groupby(['year', 'month']).count()
    monthlyData = np.empty(
        (monthly.shape[0], data.shape[1], data.shape[2], data.shape[3]))
    for month in range(0, monthly.shape[0]):
        findArr = (datesTime['year'] == monthly.index[month][0]) & \
            (datesTime['month'] == monthly.index[month][1])
        monthlyData[month, :, :, :] = data[findArr, :, :, :].mean(axis=0)

    return monthlyData


def yearlyAverage(datesTime, data):
    if len(data.shape) > 3:
        yearly = datesTime.groupby(['year']).count()
        yearlyData = np.empty(
            (yearly.shape[0], data.shape[1], data.shape[2], data.shape[3]))
        for year in range(0, yearly.shape[0]):
            if yearly.shape[0] > 1:
                findArr = (datesTime['year'] == yearly.index[year])
            else:
                findArr = (datesTime['year'] == yearly.index[year])
            yearlyData[year, :, :, :] = data[findArr, :, :, :].mean(axis=0)
    else:
        yearly = datesTime.groupby(['year']).count()
        yearlyData = np.empty((yearly.shape[0], data.shape[1], data.shape[2]))
        for year in range(0, yearly.shape[0]):
            if yearly.shape[0] > 1:
                findArr = (datesTime['year'] == yearly.index[year])
            else:
                findArr = (datesTime['year'] == yearly.index[year])
            yearlyData[year, :, :] = data[findArr, :, :].mean(axis=0)
    return yearlyData


def yearlySum(datesTime, data):
    if len(data.shape) > 3:
        yearly = datesTime.groupby(['year']).count()
        yearlyData = np.empty(
            (yearly.shape[0], data.shape[1], data.shape[2], data.shape[3]))
        for year in range(0, yearly.shape[0]):
            if yearly.shape[0] > 1:
                findArr = (datesTime['year'] == yearly.index[year])
            else:
                findArr = (datesTime['year'] == yearly.index[year])
            yearlyData[year, :, :, :] = data[findArr, :, :, :].sum(axis=0)
    else:
        yearly = datesTime.groupby(['year']).count()
        yearlyData = np.empty((yearly.shape[0], data.shape[1], data.shape[2]))
        for year in range(0, yearly.shape[0]):
            if yearly.shape[0] > 1:
                findArr = (datesTime['year'] == yearly.index[year])
            else:
                findArr = (datesTime['year'] == yearly.index[year])
            yearlyData[year, :, :] = data[findArr, :, :].sum(axis=0)

    return yearlyData


def movingAverage(datesTime, data, w):
    daily = datesTime.groupby(['year', 'month', 'day']).count()
    mvAveData = np.empty(
        (daily.shape[0], data.shape[1], data.shape[2], data.shape[3]))
    for day in range(0, daily.shape[0]):
        findArr = (datesTime['year'] == daily.index[day][0]) & \
            (datesTime['month'] == daily.index[day][1]) & \
            (datesTime['day'] == daily.index[day][2])
        for ii in range(0, findArr.sum()):
            ddData = data[findArr, :, :, :]
            if w+ii <= findArr.sum():
                dataN = ddData[ii:w+ii, :, :, :].mean(axis=0)
                if ii == 0:
                    movData = dataN
                else:
                    movData = np.max([movData, dataN], axis=0)
        mvAveData[day, :, :, :] = movData
    return mvAveData


def datePrepCMAQ(ds):
    tf = np.array(ds['TFLAG'][:])
    print(tf)
    date = []
    for ii in range(0, tf.shape[0]):
        date.append(datetime.strptime(tf.astype(str)[
                    ii], '%Y%m%d%H').strftime('%Y-%m-%d %H:00:00'))

    date = np.array(date, dtype='datetime64[s]')
    dates = pd.DatetimeIndex(date)
    datesTime = pd.DataFrame()
    datesTime['year'] = dates.year
    datesTime['month'] = dates.month
    datesTime['day'] = dates.day
    datesTime['hour'] = dates.hour
    datesTime['datetime'] = dates
    return datesTime

# def datePrepWRF(ds):
#     date = np.array(wrf.g_times.get_times(ds,timeidx=wrf.ALL_TIMES))
#     dates = pd.DatetimeIndex(date)
#     datesTime=pd.DataFrame()
#     datesTime['year'] = dates.year
#     datesTime['month'] = dates.month
#     datesTime['day'] = dates.day
#     datesTime['hour'] = dates.hour
#     datesTime['datetime']=dates
#     return datesTime


def ioapiCoords(ds):
    # Latlon
    lonI = ds.XORIG
    latI = ds.YORIG

    # Cell spacing
    xcell = ds.XCELL
    ycell = ds.YCELL
    ncols = ds.NCOLS
    nrows = ds.NROWS

    lon = np.arange(lonI, (lonI+ncols*xcell), xcell)
    lat = np.arange(latI, (latI+nrows*ycell), ycell)

    xv, yv = np.meshgrid(lon, lat)
    return xv, yv, lon, lat


def exceedance(data, criteria):
    freqExcd = np.sum(data > criteria, axis=0)
    dataTreshold = data.copy()
    dataTreshold[dataTreshold < criteria] = np.nan
    dataBin = data.copy()
    dataBin[dataBin < criteria] = 0
    dataBin[dataBin > criteria] = 1

    return freqExcd, dataTreshold, dataBin


def exceedanceSinc(dataBin, xlon, ylat):
    dataV = np.argwhere(np.sum(dataBin[:, 0, :, :], axis=0) > 0)
    dataTr = []
    for ii in dataV:
        print(ii)
        dataTr.append(dataBin[:, 0, ii[0], ii[1]])
    dataTrArr = np.array(dataTr)

    con = []
    conXY=[]
    for ii in range(0, dataTrArr.shape[0]):
        print(ii)
        for jj in range(0, dataTrArr.shape[0]):
            corrmat = np.corrcoef(dataTrArr[ii, :], dataTrArr[jj, :])
            if ii!=jj:
                if corrmat[0, 1] > 0.7:
                    con.append([dataV[ii,0],dataV[ii,1], dataV[jj,0],dataV[jj,1]])
                    conXY.append([xlon[dataV[ii,0],dataV[ii,1]],
                                 ylat[dataV[ii,0],dataV[ii,1]], 
                                 xlon[dataV[jj,0],dataV[jj,1]],
                                 ylat[dataV[jj,0],dataV[jj,1]]])
                    
    return conXY




def eqmerc2latlon(ds, xv, yv):

    mapstr = '+proj=merc +a=%s +b=%s +lat_ts=0 +lon_0=%s' % (
        6370000, 6370000, ds.XCENT)
    # p = pyproj.Proj("+proj=merc +lon_0="+str(ds.P_GAM)+" +k=1 +x_0=0 +y_0=0 +a=6370000 +b=6370000 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs")
    p = pyproj.Proj(mapstr)
    xlon, ylat = p(xv, yv, inverse=True)

    return xlon, ylat


def trimBorders(data, xv, yv, left, right, top, bottom):
    if np.size(data.shape) == 4:
        dataT = data[:, :, bottom:(data.shape[2]-top),
                     left:(data.shape[3]-right)]
        xvT = xv[bottom:(data.shape[2]-top), left:(data.shape[3]-right)]
        yvT = yv[bottom:(data.shape[2]-top), left:(data.shape[3]-right)]
    if np.size(data.shape) == 3:
        dataT = data[:, bottom:(data.shape[1]-top), left:(data.shape[2]-right)]
        xvT = xv[bottom:(data.shape[1]-top), left:(data.shape[2]-right)]
        yvT = yv[bottom:(data.shape[1]-top), left:(data.shape[2]-right)]
    if np.size(data.shape) == 2:
        dataT = data[bottom:(data.shape[0]-top), left:(data.shape[1]-right)]
        xvT = xv[bottom:(data.shape[0]-top), left:(data.shape[1]-right)]
        yvT = yv[bottom:(data.shape[0]-top), left:(data.shape[1]-right)]

    # xvT = xv[bottom:(data.shape[2]-top),left:(data.shape[3]-right)]
    # yvT = yv[bottom:(data.shape[2]-top),left:(data.shape[3]-right)]

    return dataT, xvT, yvT


def getTime(ds, data):
    dd = datePrepCMAQ(ds)
    idx2Remove = np.array(dd.drop_duplicates().index)
    data = data[idx2Remove]
    datesTime = dd.drop_duplicates().reset_index(drop=True)
    return datesTime, data

# def getTimeWRF(ds,data):
#     dd = datePrepWRF(ds)
#     idx2Remove = np.array(dd.drop_duplicates().index)
#     data = data[idx2Remove]
#     datesTime = dd.drop_duplicates().reset_index(drop=True)
#     return datesTime,data


def citiesINdomain(xlon, ylat, cities):
    s = gpd.GeoSeries(map(Point, zip(xlon.flatten(), ylat.flatten())))
    s = gpd.GeoDataFrame(geometry=s)
    s.crs = "EPSG:4326"
    s.to_crs("EPSG:4326")
    pointIn = cities.geometry.clip(s).explode()
    pointIn = gpd.GeoDataFrame({'geometry': pointIn}).reset_index()
    lia, loc = ismember(np.array((s.geometry.x, s.geometry.y)).transpose(),
                        np.array((pointIn.geometry.x, pointIn.geometry.y)).transpose(), 'rows')
    s['city'] = np.nan
    s.iloc[lia, 1] = cities['CD_MUN'][pointIn['level_0'][loc]].values
    cityMat = np.reshape(
        np.array(s.city), (xlon.shape[0], xlon.shape[1])).astype(float)
    return s, cityMat


def dataINcity(aveData, datesTime, cityMat, s, IBGE_CODE):
    # IBGE_CODE=4202404
    if np.size(aveData.shape) == 4:
        cityData = aveData[:, :, cityMat == IBGE_CODE]
        cityDataPoints = s[s.city.astype(float) == IBGE_CODE]
        cityData = cityData[:, 0, :]
        matData = aveData.copy()
        matData[:, :, cityMat != IBGE_CODE] = np.nan
        cityDataFrame = pd.DataFrame(cityData)
        cityDataFrame.columns = cityDataPoints.geometry.astype(str)
        cityDataFrame['Datetime'] = datesTime.datetime
        cityDataFrame = cityDataFrame.set_index(['Datetime'])
    else:
        cityData = aveData[:, cityMat == IBGE_CODE]
        cityDataPoints = s[s.city.astype(float) == IBGE_CODE]
        cityData = cityData[:, :]
        matData = aveData.copy()
        matData[:, cityMat != IBGE_CODE] = np.nan
        cityDataFrame = pd.DataFrame(cityData)
        cityDataFrame.columns = cityDataPoints.geometry.astype(str)
        cityDataFrame['Datetime'] = datesTime.datetime
        cityDataFrame = cityDataFrame.set_index(['Datetime'])
    return cityData, cityDataPoints, cityDataFrame, matData


def kmeanClustering(aveData,xlon,ylat,borderShape):
    
    #https://annefou.github.io/metos_python/05-scipy/
    import netCDF4
    import numpy as np
    from scipy.cluster.vq import vq,kmeans
    from matplotlib import colors as c
    import matplotlib.pyplot as plt

    lats = ylat
    lons = xlon
    pw = aveData[200,0,:,:]

    # Flatten image to get line of values
    flatraster = pw.flatten()
    #flatraster.mask = False
    flatraster = flatraster.data
    
    # Create figure to receive results
    fig = plt.figure(figsize=[20,7])
    fig.suptitle('K-Means Clustering')
    
    # In first subplot add original image
    ax = plt.subplot(241)
    ax.axis('off')
    ax.set_title('Original Image\nMonthly Average Precipitable Water\n over Ice-Free Oceans (kg m-2)')
    original=ax.pcolor(xlon,ylat,pw, cmap='rainbow_r')
    plt.colorbar(original, cmap='rainbow_r', ax=ax, orientation='vertical')
    br = gpd.read_file(borderShape)

    br.boundary.plot(edgecolor='black',linewidth=0.5,ax=ax)
    ax.set_xlim([xlon.min(), xlon.max()])
    ax.set_ylim([ylat.min(), ylat.max()]) 
    ax.set_xticks([])
    ax.set_yticks([])
    # In remaining subplots add k-means clustered images
    # Define colormap
    list_colors=['blue','orange', 'green', 'magenta', 'cyan', 'gray', 'red', 'yellow']
    for i in range(7):
        print("Calculate k-means with ", i+2, " clusters.")
        
        #This scipy code clusters k-mean, code has same length as flattened
        # raster and defines which cluster the value corresponds to
        centroids, variance = kmeans(flatraster, i+2)
        code, distance = vq(flatraster, centroids)
        
        #Since code contains the clustered values, reshape into SAR dimensions
        codeim = code.reshape(pw.shape[0], pw.shape[1])
        
        #Plot the subplot with (i+2)th k-means
        ax = plt.subplot(2,4,i+2)
        ax.axis('off')
        xlabel = str(i+2) , ' clusters'
        ax.set_title(xlabel)
        bounds=range(0,i+2)
        cmap = c.ListedColormap(list_colors[0:i+2])
        #kmp=ax.imshow(xlon,ylat,codeim, interpolation='nearest', aspect='auto', cmap=cmap,  origin='lower')
        kmp=ax.pcolor(xlon,ylat,codeim)
        plt.colorbar(kmp, cmap=cmap,  ticks=bounds, ax=ax, orientation='vertical')

        br.boundary.plot(edgecolor='black',linewidth=0.5,ax=ax)
        ax.set_xlim([xlon.min(), xlon.max()])
        ax.set_ylim([ylat.min(), ylat.max()]) 
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

def spatialCluster(aveData,xlon,ylat,datesTime):
    #https://cgc-tutorial.readthedocs.io/en/latest/notebooks/triclustering.html
    #import cgc
    #import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr
    
    #aveData = dataBin.copy()
    from cgc.triclustering import Triclustering
    from cgc.kmeans import KMeans

    spring_indices = xr.DataArray(
        data=aveData[:,:,:,:],
        dims=["time","spring_index","y", "x"],
        coords=dict(
            x=(["x"], xlon[0,:]),
            y=(["y"], ylat[:,0]),
            time=datesTime.datetime,
            spring_index=['O3'],
        ),
        attrs=dict(
            description="concentration",
            #units="degC",
        ),
    )
        


    print(spring_indices)

    
    spring_indices = spring_indices.stack(space=['x', 'y'])
    location = np.arange(spring_indices.space.size) # create a combined (x,y) index
    spring_indices = spring_indices.assign_coords(location=('space', location))
    
    # drop pixels that are null-valued for any year/spring-index
    #spring_indices = spring_indices.dropna('space', how='any')  
    print(spring_indices)


    num_band_clusters = 1
    num_time_clusters = 6
    num_space_clusters = 10
    
    max_iterations = 50  # maximum number of iterations
    conv_threshold = 0.1  # convergence threshold 
    nruns = 3  # number of differently-initialized runs
    
    tc = Triclustering(
        spring_indices.data,  # data array with shape: (bands, rows, columns)
        num_time_clusters,  
        num_space_clusters,
        num_band_clusters, 
        max_iterations=max_iterations,  
        conv_threshold=conv_threshold, 
        nruns=nruns
    )
    
    results = tc.run_with_threads(nthreads=1)
 
    time_clusters = xr.DataArray(results.bnd_clusters, dims='time', 
                                 coords=spring_indices.time.coords, 
                                 name='time cluster')
    space_clusters = xr.DataArray(results.col_clusters, dims='space', 
                                  coords=spring_indices.space.coords, 
                                  name='space cluster')
    band_clusters = xr.DataArray(results.row_clusters, dims='spring_index', 
                                 coords=spring_indices.spring_index.coords, 
                                 name='band cluster')
    
    space_clusters_xy = space_clusters.unstack('space')


    clusters = (results.bnd_clusters, results.row_clusters, results.col_clusters)
    nclusters = (num_band_clusters, num_time_clusters, num_space_clusters)
    km = KMeans(
        spring_indices.data,
        clusters=clusters,
        nclusters=nclusters,
        #k_range=range(2, 10)
    )
    results_kmeans = km.compute()
    print(f"Optimal k value: {results_kmeans.k_value}")
    
    labels = xr.DataArray(
    results_kmeans.labels, 
        coords=(
            ('band_clusters', range(num_band_clusters)),
            ('time_clusters', range(num_time_clusters)), 
            ('space_clusters', range(num_space_clusters))
        )
    )
    
    # drop tri-clusters that are not populated  
    labels = labels.dropna('band_clusters', how='all')
    labels = labels.dropna('time_clusters', how='all')
    labels = labels.dropna('space_clusters', how='all')
    labels = labels.sel(space_clusters=space_clusters, drop=True)
    labels = labels.unstack('space')
    
    
    return labels,space_clusters_xy,time_clusters