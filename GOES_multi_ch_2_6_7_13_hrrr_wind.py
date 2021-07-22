#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 00:43:03 2021

@author: matthewroberts
"""
from boto3.session import Session
import numpy as np
import numpy.ma as ma
import sys
from netCDF4 import Dataset
import pygrib
import datetime as dt
import pandas as pd
import os
import glob

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

from cpt_convert import loadCPT # Import the CPT convert function

import matplotlib
from matplotlib import path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import warnings
warnings.filterwarnings('ignore')

start_time = dt.datetime.now()
print('\nProgram Started: {0}'.format(start_time))
print('===================')

##########################################
# Inputs
##########################################
sat = 'goes17' #goes16/17
product = 'ABI-L2-CMIP'
domain = 'M1' # F, C, M1/M2
band1 = '02' # (visible)
band2 = '13' # (IR)
band3 = '06' # (IR)
band4 = '07' # (SWIR)

# Set the start and end date for the files we want to download
# Make sure to have a difference of at least 15-20 min between
# sDate and eDate otherwise things get weird.
sDATE = dt.datetime(2021, 7, 20, 23, 0)
eDATE = dt.datetime(2021, 7, 21, 0, 0)

keyword = 'Fire_name' # For naming save directory

labels = False # Set lat/lon labels on axes
center_lon = -120.98
center_lat = 40.26
zoom = 1.1 # 1=closest, larger=further

# Set plotting area
extent = [center_lon-.7*zoom, center_lon+.7*zoom, # leftlon,rightlon
          center_lat-.4*zoom, center_lat+.4*zoom] # lowerlat,upperlat

# Setup directories
mainpath = '/Projects/GOES_Tools'
# Create folders in mainpath
hrrrpath = 'hrrr_files'
goespath = 'netcdf_files'
savepath = 'images'
# Set path to county shapefiles for plotting later
shapefile_path = '/countyl010g_shp/countyl010g.shp'
# Path to custom cmaps
cmappath = '/cmaps'

# AWS credentials
ACCESS_KEY='xxxxxxxxxxxx'
SECRET_KEY='xxxxxxxxxxxx'

##########################################
# Set up functions
##########################################
def latlon_convert(proj_info,lat_rad_1d,lon_rad_1d):
    """
    Ensures all projection info is consistent with lat/lon map.
    """
    lon_origin = proj_info.longitude_of_projection_origin
    H = proj_info.perspective_point_height+proj_info.semi_major_axis
    r_eq = proj_info.semi_major_axis
    r_pol = proj_info.semi_minor_axis

    # create meshgrid filled with radian angles
    lat_rad,lon_rad = np.meshgrid(lat_rad_1d,lon_rad_1d)

    # lat/lon calc routine from satellite radian angle vectors
    lambda_0 = (lon_origin*np.pi)/180.0

    a_var = np.power(np.sin(lat_rad),2.0) + (np.power(np.cos(lat_rad),2.0)*(np.power(np.cos(lon_rad),2.0)+(((r_eq*r_eq)/(r_pol*r_pol))*np.power(np.sin(lon_rad),2.0))))
    b_var = -2.0*H*np.cos(lat_rad)*np.cos(lon_rad)
    c_var = (H**2.0)-(r_eq**2.0)

    r_s = (-1.0*b_var - np.sqrt((b_var**2)-(4.0*a_var*c_var)))/(2.0*a_var)

    s_x = r_s*np.cos(lat_rad)*np.cos(lon_rad)
    s_y = - r_s*np.sin(lat_rad)
    s_z = r_s*np.cos(lat_rad)*np.sin(lon_rad)

    # latitude and longitude projection for plotting data on traditional lat/lon maps
    lat = (180.0/np.pi)*(np.arctan(((r_eq*r_eq)/(r_pol*r_pol))*((s_z/np.sqrt(((H-s_x)*(H-s_x))+(s_y*s_y))))))
    lon = (lambda_0 - np.arctan(s_y/(H-s_x)))*(180.0/np.pi)

    return lat, lon

def printScanInfo(ds,bandname):
    """
    Prints some info about file/scan/channel for verification.
    """
    # Scan's start time, converted to datetime object
    scan_start = dt.datetime.strptime(ds.time_coverage_start, '%Y-%m-%dT%H:%M:%S.%fZ')
    # Scan's end time, converted to datetime object
    scan_end = dt.datetime.strptime(ds.time_coverage_end, '%Y-%m-%dT%H:%M:%S.%fZ')
    # File creation time, convert to datetime object
    file_created = dt.datetime.strptime(ds.date_created, '%Y-%m-%dT%H:%M:%S.%fZ')

    # The 't' variable is the scan's midpoint time
    midpoint = float(ds.variables['t'][:])
    scan_mid = dt.datetime(2000,1,1,12) + dt.timedelta(seconds=midpoint)

    band_wl = str(ds.variables[bandname][0])

    # print('------------------------------------')
    # print('Band          : {} um'.format(band_wl))
    # print('Scan Start    : {}'.format(scan_start))
    # print('Scan midpoint : {}'.format(scan_mid))
    # print('Scan End      : {}'.format(scan_end))
    # print('File Created  : {}'.format(file_created))
    # print('Scan Duration : {:.2f} minutes'.format((scan_end-scan_start).seconds/60))

    return scan_mid, band_wl

def bbox2ij(lon,lat,bbox=[-160., -155., 18., 23.]):
    """Return indices for i,j that will completely cover the specified bounding box.
    i0,i1,j0,j1 = bbox2ij(lon,lat,bbox)
    lon,lat = 2D arrays that are the target of the subset
    bbox = list containing the bounding box: [lon_min, lon_max, lat_min, lat_max]

    Example
    -------
    >>> i0,i1,j0,j1 = bbox2ij(lon_rho,[-71, -63., 39., 46])
    >>> h_subset = nc.variables['h'][j0:j1,i0:i1]
    """
    bbox = np.array(bbox)
    mypath = np.array([bbox[[0,1,1,0]],bbox[[2,2,3,3]]]).T
    p = path.Path(mypath)
    points = np.vstack((lon.flatten(),lat.flatten())).T
    n,m = np.shape(lon)
    inside = p.contains_points(points).reshape((n,m))
    ii,jj = np.meshgrid(range(m),range(n))
    return min(ii[inside]),max(ii[inside]),min(jj[inside]),max(jj[inside])

def showSpines(axis):
    axis.spines['top'].set_visible(True)
    axis.spines['bottom'].set_visible(True)
    axis.spines['left'].set_visible(True)
    axis.spines['right'].set_visible(True)
    axis.spines['top'].set_linewidth(3)
    axis.spines['bottom'].set_linewidth(3)
    axis.spines['left'].set_linewidth(3)
    axis.spines['right'].set_linewidth(3)
    axis.spines['top'].set_color('k')
    axis.spines['bottom'].set_color('k')
    axis.spines['left'].set_color('k')
    axis.spines['right'].set_color('k')

def showLabels(axis,labs):
    gl = axis.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0, alpha=0.0)
    gl.top_labels = False
    gl.bottom_labels = labs
    gl.left_labels = False
    gl.right_labels = False
    gl.xlines = False
    gl.ylines = False

##########################################
# Fail safes
##########################################
if (sDATE > eDATE):
    sys.exit('*** Start date must be before end date! ***')

if ((eDATE - sDATE) < dt.timedelta(seconds=900)):
    sys.exit('*** Start and end dates need to be at least 15 min apart! ***')

if (start_time < eDATE):
    sys.exit('*** Time period must be before current time! ***')

##########################################
# Custom colormap setup
##########################################
# Converts the CPT file to be used in Python
cpt = loadCPT(cmappath+'/IR4AVHRR6.cpt')
cpt2 = loadCPT(cmappath+'/red.cpt')
cpt3 = loadCPT(cmappath+'/sst.cpt')
# Makes a linear interpolation with the CPT file
cpt_convert = LinearSegmentedColormap('cpt', cpt)
cpt_red = LinearSegmentedColormap('cpt2', cpt2)
cpt_sst = LinearSegmentedColormap('cpt3', cpt3)

##########################################
# Set up datetime arrays
##########################################
print('\nStart: {0} UTC'.format(sDATE))
print('End: {0} UTC'.format(eDATE))
# Create a list of datetimes we want to download with Pandas `date_range` function.
# The HRRR model is run every hour, so make a list of every hour
DATES = pd.date_range(sDATE, eDATE, freq='5min')
DATES = [d.to_pydatetime() for d in DATES]
# Create hourly timestamps from 15min
ymdh = []
for d in DATES:
    ymdh.append(str(d.year)+str(d.month).zfill(2)+str(d.day).zfill(2)+'_'+str(d.hour).zfill(2))
ymdh = list(set(ymdh))

_DATES = []
for d in ymdh:
    _DATES.append(dt.datetime.strptime(d,'%Y%m%d_%H'))
_DATES = sorted(_DATES)

##########################################
# Open AWS session for HRRR download
##########################################
# Remove any old hrrr files
files = glob.glob(mainpath+'/'+hrrrpath+'/*')
for f in files:
    os.remove(f)

session = Session(aws_access_key_id=ACCESS_KEY,
                  aws_secret_access_key=SECRET_KEY)
s3 = session.client('s3')

bucket = 'noaa-hrrr-bdp-pds'
_filelist = []
for d in _DATES:
    prefix = 'hrrr.'+dt.datetime.strftime(d,'%Y%m%d')+'/conus/hrrr.t'+str(d.hour).zfill(2)+'z.wrfsubhf'
    if (s3.list_objects_v2(Bucket=bucket, Prefix=prefix)['KeyCount'] > 2):
        file_obj = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)['Contents']
        for f in file_obj:
            if (f['Key'].endswith('f00.grib2') or f['Key'].endswith('f01.grib2')):
                _filelist.append(f['Key'])
    else:
        d = d - dt.timedelta(hours=1)
        prefix = 'hrrr.'+dt.datetime.strftime(d,'%Y%m%d')+'/conus/hrrr.t'+str(d.hour).zfill(2)+'z.wrfsubhf'
        file_obj = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)['Contents']
        for f in file_obj:
            if (f['Key'].endswith('f01.grib2') or f['Key'].endswith('f02.grib2')):
                _filelist.append(f['Key'])

filelist = []
filekeys = []
for d in DATES:
    if ((d.minute < 15) & ('hrrr.'+dt.datetime.strftime(d,'%Y%m%d')+'/conus/hrrr.t'+str(d.hour).zfill(2)+'z.wrfsubhf00.grib2' in _filelist)):
        filelist.append('hrrr.'+dt.datetime.strftime(d,'%Y%m%d')+'/conus/hrrr.t'+str(d.hour).zfill(2)+'z.wrfsubhf00.grib2')
        filekeys.append('hrrr.'+dt.datetime.strftime(d,'%Y%m%d')+'_'+str(d.hour).zfill(2)+'00_t'+str(d.hour).zfill(2)+'zf00')
    elif (d.minute < 15):
        d0 = d
        d = d - dt.timedelta(hours=1)
        filelist.append('hrrr.'+dt.datetime.strftime(d,'%Y%m%d')+'/conus/hrrr.t'+str(d.hour).zfill(2)+'z.wrfsubhf01.grib2')
        filekeys.append('hrrr.'+dt.datetime.strftime(d,'%Y%m%d')+'_'+str(d0.hour).zfill(2)+'00_t'+str(d.hour).zfill(2)+'zf01')

    if ((d.minute >= 15) & (d.minute < 60) & ('hrrr.'+dt.datetime.strftime(d,'%Y%m%d')+'/conus/hrrr.t'+str(d.hour).zfill(2)+'z.wrfsubhf01.grib2' in _filelist)):
        filelist.append('hrrr.'+dt.datetime.strftime(d,'%Y%m%d')+'/conus/hrrr.t'+str(d.hour).zfill(2)+'z.wrfsubhf01.grib2')
        if ((d.minute >= 15) & (d.minute < 30)):
            filekeys.append('hrrr.'+dt.datetime.strftime(d,'%Y%m%d')+'_'+str(d.hour).zfill(2)+'15_t'+str(d.hour).zfill(2)+'zf01')
        elif ((d.minute >= 30) & (d.minute < 45)):
            filekeys.append('hrrr.'+dt.datetime.strftime(d,'%Y%m%d')+'_'+str(d.hour).zfill(2)+'30_t'+str(d.hour).zfill(2)+'zf01')
        elif ((d.minute >= 45) & (d.minute < 60)):
            filekeys.append('hrrr.'+dt.datetime.strftime(d,'%Y%m%d')+'_'+str(d.hour).zfill(2)+'45_t'+str(d.hour).zfill(2)+'zf01')
    elif ((d.minute >= 15) & (d.minute < 60)):
        d0 = d
        d = d - dt.timedelta(hours=1)
        filelist.append('hrrr.'+dt.datetime.strftime(d,'%Y%m%d')+'/conus/hrrr.t'+str(d.hour).zfill(2)+'z.wrfsubhf02.grib2')
        if ((d.minute >= 15) & (d.minute < 30)):
            filekeys.append('hrrr.'+dt.datetime.strftime(d,'%Y%m%d')+'_'+str(d0.hour).zfill(2)+'15_t'+str(d.hour).zfill(2)+'zf02')
        elif ((d.minute >= 30) & (d.minute < 45)):
            filekeys.append('hrrr.'+dt.datetime.strftime(d,'%Y%m%d')+'_'+str(d0.hour).zfill(2)+'30_t'+str(d.hour).zfill(2)+'zf02')
        elif ((d.minute >= 45) & (d.minute < 60)):
            filekeys.append('hrrr.'+dt.datetime.strftime(d,'%Y%m%d')+'_'+str(d0.hour).zfill(2)+'45_t'+str(d.hour).zfill(2)+'zf02')

# Remove any duplicate file downloads
_filelist = list(dict.fromkeys(filelist))
filekeys = list(dict.fromkeys(filekeys))

print('\nHRRR files to download: \n'+str(_filelist))
print('\nDownloading from AWS...')
for f in _filelist:
    s3.download_file(bucket, f, mainpath+'/'+hrrrpath+'/'+f[5:13]+'.'+os.path.basename(f))
    print(f[5:13]+'.'+os.path.basename(f)+' downloaded...')
#%%
##########################################
# Build HRRR data arrays
##########################################
newdates = []
for k in range(len(filekeys)):
    gribfile = filekeys[k][5:13]+'.hrrr.'+filekeys[k][-7:-3]+'.wrfsubh'+filekeys[k][-3:]+'.grib2'
    ftype = filekeys[k][-3:]
    d = dt.datetime.strptime(filekeys[k][5:18],'%Y%m%d_%H%M')

    if (ftype == 'f00'):
        mins1 = '0' # corresponds w/ 0 mins/f00. Maybe a typo in the hrrr grib files
    elif (ftype == 'f01'):
        mins1 = '1' # corresponds w/ 60 mins/f01. Maybe a typo in the hrrr grib files
        mins2 = '15'
        mins3 = '30'
        mins4 = '45'
    elif (ftype == 'f02'):
        mins1 = '1' # corresponds w/ 120 mins/f02. Maybe a typo in the hrrr grib files
        mins2 = '75'
        mins3 = '90'
        mins4 = '105'

    if (d.minute == 0):
        gr = pygrib.open(mainpath+'/'+hrrrpath+'/'+gribfile)
        # Find 10m winds in grib file
        grb = gr.select(name='10 metre U wind component')[:]
        # Convert list items to strings
        grb = [str(g) for g in grb]
        # Find 00, 15, 30, 45 min fcast strings, then extract message number, then convert back to int
        if (mins1 == '0'):
            idx00 = int([s for s in grb if "fcst time "+mins1+" mins" in s][0][:2])
        elif (mins1 == '1'):
            idx00 = int([s for s in grb if "fcst time "+mins1+" mins" in s][0][:3])
        # Get u, v values
        u = gr[idx00].values
        v = gr[idx00+1].values
        lats,lons = gr[idx00].latlons()
        gr.close()
    elif (d.minute == 15):
        gr = pygrib.open(mainpath+'/'+hrrrpath+'/'+gribfile)
        # Find 10m winds in grib file
        grb = gr.select(name='10 metre U wind component')[:]
        # Convert list items to strings
        grb = [str(g) for g in grb]
        # Find 00, 15, 30, 45 min fcast strings, then extract message number, then convert back to int
        idx15 = int([s for s in grb if "fcst time "+mins2+" mins" in s][0][:2])
        # Get u, v values
        u = gr[idx15].values
        v = gr[idx15+1].values
        lats,lons = gr[idx15].latlons()
        gr.close()
    elif (d.minute == 30):
        gr = pygrib.open(mainpath+'/'+hrrrpath+'/'+gribfile)
        # Find 10m winds in grib file
        grb = gr.select(name='10 metre U wind component')[:]
        # Convert list items to strings
        grb = [str(g) for g in grb]
        # Find 00, 15, 30, 45 min fcast strings, then extract message number, then convert back to int
        idx30 = int([s for s in grb if "fcst time "+mins3+" mins" in s][0][:2])
        # Get u, v values
        u = gr[idx30].values
        v = gr[idx30+1].values
        lats,lons = gr[idx30].latlons()
        gr.close()
    elif (d.minute == 45):
        gr = pygrib.open(mainpath+'/'+hrrrpath+'/'+gribfile)
        # Find 10m winds in grib file
        grb = gr.select(name='10 metre U wind component')[:]
        # Convert list items to strings
        grb = [str(g) for g in grb]
        # Find 00, 15, 30, 45 min fcast strings, then extract message number, then convert back to int
        idx45 = int([s for s in grb if "fcst time "+mins4+" mins" in s][0][:3]) #search first 3 indices because likely >100
        # Get u, v values
        u = gr[idx45].values
        v = gr[idx45+1].values
        lats,lons = gr[idx45].latlons()
        gr.close()
    if (k == 0):
        _u = u
        _v = v
    else:
        _u = np.dstack([_u,u])
        _v = np.dstack([_v,v])
    newdates.append(d)

# Consolidate arrays for plotting
hrrr_t = np.asarray(newdates)
hrrr_x = lons
hrrr_y = lats
hrrr_u = _u
hrrr_v = _v

#%%
##########################################
# Open AWS session for GOES download
##########################################
print('\nFetching GOES Files...')
print('Downloading from AWS...')
count = 0
for d in _DATES:
    bucket = 'noaa-'+sat

    if ((domain == 'M1') or (domain == 'M2')):
        prefix1 = product+'M/'+str(d.year)+'/'+str(d.timetuple().tm_yday)+'/'+str(d.hour).zfill(2)+'/OR_'+product+domain+'-M6C'+band1
        prefix2 = product+'M/'+str(d.year)+'/'+str(d.timetuple().tm_yday)+'/'+str(d.hour).zfill(2)+'/OR_'+product+domain+'-M6C'+band2
        prefix3 = product+'M/'+str(d.year)+'/'+str(d.timetuple().tm_yday)+'/'+str(d.hour).zfill(2)+'/OR_'+product+domain+'-M6C'+band3
        prefix4 = product+'M/'+str(d.year)+'/'+str(d.timetuple().tm_yday)+'/'+str(d.hour).zfill(2)+'/OR_'+product+domain+'-M6C'+band4
    else:
        prefix1 = product+domain+'/'+str(d.year)+'/'+str(d.timetuple().tm_yday)+'/'+str(d.hour).zfill(2)+'/OR_'+product+domain+'-M6C'+band1
        prefix2 = product+domain+'/'+str(d.year)+'/'+str(d.timetuple().tm_yday)+'/'+str(d.hour).zfill(2)+'/OR_'+product+domain+'-M6C'+band2
        prefix3 = product+domain+'/'+str(d.year)+'/'+str(d.timetuple().tm_yday)+'/'+str(d.hour).zfill(2)+'/OR_'+product+domain+'-M6C'+band3
        prefix4 = product+domain+'/'+str(d.year)+'/'+str(d.timetuple().tm_yday)+'/'+str(d.hour).zfill(2)+'/OR_'+product+domain+'-M6C'+band4

    filelist1 = s3.list_objects_v2(Bucket=bucket, Prefix=prefix1)['Contents']
    filelist2 = s3.list_objects_v2(Bucket=bucket, Prefix=prefix2)['Contents']
    filelist3 = s3.list_objects_v2(Bucket=bucket, Prefix=prefix3)['Contents']
    filelist4 = s3.list_objects_v2(Bucket=bucket, Prefix=prefix4)['Contents']

    for key in range(len(filelist1)):
        # Make sure files fall within desired date/time range
        tstamp = dt.datetime.strptime(filelist1[key]['Key'][-17:-6],'%Y%j%H%M')

        if (tstamp >= sDATE) & (tstamp <= eDATE):
            s3.download_file(bucket, filelist1[key]['Key'], mainpath+'/'+goespath+'/hrrr_satfile1.nc')
            s3.download_file(bucket, filelist2[key]['Key'], mainpath+'/'+goespath+'/hrrr_satfile2.nc')
            s3.download_file(bucket, filelist3[key]['Key'], mainpath+'/'+goespath+'/hrrr_satfile3.nc')
            s3.download_file(bucket, filelist4[key]['Key'], mainpath+'/'+goespath+'/hrrr_satfile4.nc')

            filename1 = mainpath+'/'+goespath+'/hrrr_satfile1.nc'
            filename2 = mainpath+'/'+goespath+'/hrrr_satfile2.nc'
            filename3 = mainpath+'/'+goespath+'/hrrr_satfile3.nc'
            filename4 = mainpath+'/'+goespath+'/hrrr_satfile4.nc'
            data1 = Dataset(filename1,'r')
            data2 = Dataset(filename2,'r')
            data3 = Dataset(filename3,'r')
            data4 = Dataset(filename4,'r')

            scan_time1,band_wl1 = printScanInfo(data1,'band_wavelength')
            scan_time2,band_wl2 = printScanInfo(data2,'band_wavelength')
            scan_time3,band_wl3 = printScanInfo(data3,'band_wavelength')
            scan_time4,band_wl4 = printScanInfo(data4,'band_wavelength')

            if (sat == 'goes17'):
                satname = 'GOES-17'
            if (sat == 'goes16'):
                satname = 'GOES-16'

            # GOES-17 projection info and retrieving relevant constants
            # Visible channels x/y grid
            projInfo = data1.variables['goes_imager_projection']
            latRad = data1.variables['x'][:]
            lonRad = data1.variables['y'][:]
            y, x = latlon_convert(projInfo,latRad,lonRad)
            # Remove GOES16 masking
            x.mask = ma.nomask
            y.mask = ma.nomask

            # IR channels x/y grid
            projInfo = data2.variables['goes_imager_projection']
            latRad = data2.variables['x'][:]
            lonRad = data2.variables['y'][:]
            y2, x2 = latlon_convert(projInfo,latRad,lonRad)
            # Remove GOES16 masking
            x2.mask = ma.nomask
            y2.mask = ma.nomask

            # Data
            goesdat1 = data1.variables['CMI'][:]
            goesdat2 = data2.variables['CMI'][:]
            goesdat3 = data3.variables['CMI'][:]
            goesdat4 = data4.variables['CMI'][:]

            # close file when finished
            data1.close()
            data2.close()
            data3.close()
            data4.close()
            data1 = None
            data2 = None
            data3 = None
            data4 = None

            print('Syncing HRRR data with GOES...')
            if (count == 0):
                _scan_time = scan_time1
                _goesdat1 = goesdat1
                _goesdat2 = goesdat2
                _goesdat3 = goesdat3
                _goesdat4 = goesdat4

                for tt in [0,15,30,45]:
                    if ((tstamp.minute >= tt) & (tstamp.minute < tt+15)):
                        idx = np.where([(h.minute >= tt) & (h.minute < tt+15) & (h.hour == tstamp.hour) for h in hrrr_t])
                        new_hrrr_t = hrrr_t[idx]
                        new_hrrr_u = hrrr_u[:,:,idx]
                        new_hrrr_v = hrrr_v[:,:,idx]
            else:
                _scan_time = np.dstack([_scan_time,scan_time1])
                _goesdat1 = np.dstack([_goesdat1,goesdat1])
                _goesdat2 = np.dstack([_goesdat2,goesdat2])
                _goesdat3 = np.dstack([_goesdat3,goesdat3])
                _goesdat4 = np.dstack([_goesdat4,goesdat4])

                for tt in [0,15,30,45]:
                    if ((tstamp.minute >= tt) & (tstamp.minute < tt+15)):
                        idx = np.where([(h.minute >= tt) & (h.minute < tt+15) & (h.hour == tstamp.hour) for h in hrrr_t])
                        new_hrrr_t = np.dstack([new_hrrr_t,hrrr_t[idx]])
                        new_hrrr_u = np.dstack([new_hrrr_u,hrrr_u[:,:,idx]])
                        new_hrrr_v = np.dstack([new_hrrr_v,hrrr_v[:,:,idx]])

            count = count+1
            print('Ch. '+band1+', '+band2+', '+band3+', '+band4+' '+dt.datetime.strftime(scan_time1,'%Y-%m-%d %H:%M')+' downloaded and synced...')

# Time synced arrays
goes_t = np.squeeze(_scan_time)
goes_x = x
goes_y = y
goes_x2 = x2
goes_y2 = y2
goes_ch2 = np.sqrt(_goesdat1) # Gamma correction for ch 02
goes_ch13 = _goesdat2
goes_ch6 = np.sqrt(_goesdat3) # Gamma correction for ch 06
goes_ch7 = _goesdat4

hrrr_t = np.squeeze(new_hrrr_t)
hrrr_x = hrrr_x
hrrr_y = hrrr_y
hrrr_u = np.squeeze(new_hrrr_u)*1.94384 # ms^-1 to kts
hrrr_v = np.squeeze(new_hrrr_v)*1.94384 # ms^-1 to kts

# Define clipping region slightly larger than mapped region to speed up plotting
extent2 = [center_lon-.7*(zoom*2), center_lon+.7*(zoom*2), # llon,rlon
           center_lat-.5*(zoom*2), center_lat+.5*(zoom*2)] # llat,ulat

i0,i1,j0,j1 = bbox2ij(hrrr_x,hrrr_y,extent2)
hrrr_x = hrrr_x[j0:j1, i0:i1]
hrrr_y = hrrr_y[j0:j1, i0:i1]
hrrr_u = hrrr_u[j0:j1, i0:i1, :]
hrrr_v = hrrr_v[j0:j1, i0:i1, :]

i0,i1,j0,j1 = bbox2ij(goes_x,goes_y,extent2)
goes_x = goes_x[j0:j1, i0:i1]
goes_y = goes_y[j0:j1, i0:i1]
goes_ch2 = goes_ch2[j0:j1, i0:i1, :]

i0,i1,j0,j1 = bbox2ij(goes_x2,goes_y2,extent2)
goes_x2 = goes_x2[j0:j1, i0:i1]
goes_y2 = goes_y2[j0:j1, i0:i1]
goes_ch6 = goes_ch6[j0:j1, i0:i1, :]
goes_ch7 = goes_ch7[j0:j1, i0:i1, :]
goes_ch13 = goes_ch13[j0:j1, i0:i1, :]

# Download shapefile for US counties
print("\nLoading Shapefiles...")
reader = shpreader.Reader(shapefile_path)
counties = list(reader.geometries())
COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())
#%%
##########################################
# Plot
##########################################
print('Plotting...')

for t in range(len(goes_t)):
    fig = plt.figure(figsize=(16, 9))
    spec = gridspec.GridSpec(nrows=18, ncols=32, figure=fig)

    ##########################################
    # Top left panel
    ##########################################
    ax1 = fig.add_subplot(spec[0:6, 6:16], projection=ccrs.PlateCarree())
    ax1.set_extent(extent)

    cmap = 'binary_r'
    pcm1 = ax1.pcolormesh(goes_x, goes_y, goes_ch2[:,:,t], cmap=cmap, vmin=0, vmax=1,
                          transform=ccrs.PlateCarree(), alpha=1, zorder=9)

    # Adjust barb density for different zoom levels
    if ((zoom >= .5) & (zoom < 1)):
        b = ax1.barbs(hrrr_x[::2,::2],hrrr_y[::2,::2],hrrr_u[::2,::2,t],hrrr_v[::2,::2,t],sizes=dict(emptybarb=0.05),zorder=10)
    elif (zoom >= 1):
        b = ax1.barbs(hrrr_x[::4,::4],hrrr_y[::4,::4],hrrr_u[::4,::4,t],hrrr_v[::4,::4,t],sizes=dict(emptybarb=0.05),zorder=10)
    else:
        b = ax1.barbs(hrrr_x,hrrr_y,hrrr_u[:,:,t],hrrr_v[:,:,t],sizes=dict(emptybarb=0.05),zorder=10)

    ax1.add_feature(COUNTIES, facecolor='none', edgecolor='k', linewidth=3, alpha=.4, zorder=10)

    ax1.set_title(satname+' Ch. '+band1+'-'+band_wl1+' um '+dt.datetime.strftime(goes_t[t],'%Y-%m-%d  %H:%M')+'Z \n'+
                  'HRRR 10m-Winds Valid: '+dt.datetime.strftime(hrrr_t[t],'%Y-%m-%d  %H:%M')+'Z ',
                  fontweight='bold', color='white', fontsize=6, y=.87,
                  loc='right', zorder=10,
                  path_effects=[PathEffects.withStroke(linewidth=2,foreground="k")])

    showLabels(ax1,labels)
    showSpines(ax1)

    # cbbox = inset_axes(ax1, '12%', '77%', loc = 'lower right')
    # [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
    # cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False,
    #                   labelleft=False, labeltop=False, labelright=False, labelbottom=False)
    # cbbox.set_facecolor([1,1,1,0.75])

    # cbaxes = inset_axes(cbbox, '30%', '84%', loc = 'lower left')
    # cbar2 = fig.colorbar(pcm1,cax=cbaxes)
    # cbar2.ax.set_title('Reflectance\n', ha='center', fontsize=12, fontweight='heavy')
    # cbar2.ax.tick_params(labelsize=13)

    ##########################################
    # Top right panel
    ##########################################
    ax2 = fig.add_subplot(spec[0:6, 16:26], projection=ccrs.PlateCarree())
    ax2.set_extent(extent)

    cmap = cpt_convert
    pcm2 = ax2.pcolormesh(goes_x2, goes_y2, goes_ch13[:,:,t], cmap=cmap, vmin=170, vmax=378,
                          transform=ccrs.PlateCarree(), alpha=1, zorder=9)

    # Adjust barb density for different zoom levels
    if ((zoom >= .5) & (zoom < 1)):
        b = ax2.barbs(hrrr_x[::2,::2],hrrr_y[::2,::2],hrrr_u[::2,::2,t],hrrr_v[::2,::2,t],sizes=dict(emptybarb=0.05),zorder=10)
    elif (zoom >= 1):
        b = ax2.barbs(hrrr_x[::4,::4],hrrr_y[::4,::4],hrrr_u[::4,::4,t],hrrr_v[::4,::4,t],sizes=dict(emptybarb=0.05),zorder=10)
    else:
        b = ax2.barbs(hrrr_x,hrrr_y,hrrr_u[:,:,t],hrrr_v[:,:,t],sizes=dict(emptybarb=0.05),zorder=10)

    ax2.add_feature(COUNTIES, facecolor='none', edgecolor='k', linewidth=3, alpha=.4, zorder=10)

    ax2.set_title(satname+' Ch. '+band2+'-'+band_wl2+' um '+dt.datetime.strftime(goes_t[t],'%Y-%m-%d  %H:%M')+'Z \n'+
                  'HRRR 10m-Winds Valid: '+dt.datetime.strftime(hrrr_t[t],'%Y-%m-%d  %H:%M')+'Z ',
                  fontweight='bold', color='white', fontsize=6, y=.87,
                  loc='right', zorder=10,
                  path_effects=[PathEffects.withStroke(linewidth=2,foreground="k")])

    showLabels(ax2,labels)
    showSpines(ax2)

    cbbox = inset_axes(ax2, '14%', '76%', loc = 'lower right')
    [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
    cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False,
                      labelleft=False, labeltop=False, labelright=False, labelbottom=False)
    cbbox.set_facecolor([1,1,1,0.7])

    cbaxes = inset_axes(cbbox, '20%', '78%', loc = 'lower left')
    cbar1 = fig.colorbar(pcm2,cax=cbaxes)
    cbar1.ax.set_title('       BT [K]', ha='center', fontsize=8, fontweight='heavy')
    cbar1.ax.tick_params(labelsize=8)

    ##########################################
    # Bottom left panel
    ##########################################
    ax3 = fig.add_subplot(spec[6:12, 6:16], projection=ccrs.PlateCarree())
    ax3.set_extent(extent)

    cmap = 'Greys_r'
    pcm3 = ax3.pcolormesh(goes_x2, goes_y2, goes_ch6[:,:,t], cmap=cmap, vmin=0, vmax=1,
                          transform=ccrs.PlateCarree(), alpha=1, zorder=9)

    # Adjust barb density for different zoom levels
    if ((zoom >= .5) & (zoom < 1)):
        b = ax3.barbs(hrrr_x[::2,::2],hrrr_y[::2,::2],hrrr_u[::2,::2,t],hrrr_v[::2,::2,t],sizes=dict(emptybarb=0.05),zorder=10)
    elif (zoom >= 1):
        b = ax3.barbs(hrrr_x[::4,::4],hrrr_y[::4,::4],hrrr_u[::4,::4,t],hrrr_v[::4,::4,t],sizes=dict(emptybarb=0.05),zorder=10)
    else:
        b = ax3.barbs(hrrr_x,hrrr_y,hrrr_u[:,:,t],hrrr_v[:,:,t],sizes=dict(emptybarb=0.05),zorder=10)

    ax3.add_feature(COUNTIES, facecolor='none', edgecolor='k', linewidth=3, alpha=.4, zorder=10)

    ax3.set_title(satname+' Ch. '+band3+'-'+band_wl3+' um '+dt.datetime.strftime(goes_t[t],'%Y-%m-%d  %H:%M')+'Z \n'+
                  'HRRR 10m-Winds Valid: '+dt.datetime.strftime(hrrr_t[t],'%Y-%m-%d  %H:%M')+'Z ',
                  fontweight='bold', color='white', fontsize=6, y=.87,
                  loc='right', zorder=10,
                  path_effects=[PathEffects.withStroke(linewidth=2,foreground="k")])

    showLabels(ax3,labels)
    showSpines(ax3)

    ##########################################
    # Bottom right panel
    ##########################################
    ax4 = fig.add_subplot(spec[6:12, 16:26], projection=ccrs.PlateCarree())
    ax4.set_extent(extent)

    cmp1 = cpt_red
    cmp1_colors = cmp1(np.linspace(0, 1, 100))[::-1] #flip cmap
    maroon_to_black = cmp1_colors[20:90, :] #snip desired section out

    cmp2 = cpt_sst
    newcolors = cmp2(np.linspace(0, 1, 256))
    newcolors[-70:, :] = maroon_to_black #paste snipped section into new cmap
    newcmp = ListedColormap(newcolors)

    cmap = newcmp
    pcm4 = ax4.pcolormesh(goes_x2, goes_y2, goes_ch7[:,:,t], cmap=cmap, vmin=195, vmax=430,
                          transform=ccrs.PlateCarree(), alpha=1, zorder=9)

    # Adjust barb density for different zoom levels
    if ((zoom >= .5) & (zoom < 1)):
        b = ax4.barbs(hrrr_x[::2,::2],hrrr_y[::2,::2],hrrr_u[::2,::2,t],hrrr_v[::2,::2,t],sizes=dict(emptybarb=0.05),zorder=10)
    elif (zoom >= 1):
        b = ax4.barbs(hrrr_x[::4,::4],hrrr_y[::4,::4],hrrr_u[::4,::4,t],hrrr_v[::4,::4,t],sizes=dict(emptybarb=0.05),zorder=10)
    else:
        b = ax4.barbs(hrrr_x,hrrr_y,hrrr_u[:,:,t],hrrr_v[:,:,t],sizes=dict(emptybarb=0.05),zorder=10)

    ax4.add_feature(COUNTIES, facecolor='none', edgecolor='k', linewidth=3, alpha=.4, zorder=10)

    ax4.set_title(satname+' Ch. '+band4+'-'+band_wl4+' um '+dt.datetime.strftime(goes_t[t],'%Y-%m-%d  %H:%M')+'Z \n'+
                  'HRRR 10m-Winds Valid: '+dt.datetime.strftime(hrrr_t[t],'%Y-%m-%d  %H:%M')+'Z ',
                  fontweight='bold', color='white', fontsize=6, y=.87,
                  loc='right', zorder=10,
                  path_effects=[PathEffects.withStroke(linewidth=2,foreground="k")])

    showLabels(ax4,labels)
    showSpines(ax4)

    cbbox = inset_axes(ax4, '14%', '76%', loc = 'lower right')
    [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
    cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False,
                      labelleft=False, labeltop=False, labelright=False, labelbottom=False)
    cbbox.set_facecolor([1,1,1,0.7])

    cbaxes = inset_axes(cbbox, '20%', '78%', loc = 'lower left')
    cbar2 = fig.colorbar(pcm4,cax=cbaxes)
    cbar2.ax.set_title('       BT [K]', ha='center', fontsize=8, fontweight='heavy')
    cbar2.ax.tick_params(labelsize=8)

    saveDir = mainpath+'/'+savepath+'/'+keyword+'_'+dt.datetime.strftime(sDATE,'%Y%m%d')
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    plt.savefig(saveDir+'/'+dt.datetime.strftime(goes_t[t],'%Y%m%d_%H%M')+'.png',bbox_inches='tight',dpi=120)
    plt.close()
    print(dt.datetime.strftime(goes_t[t],'%Y%m%d %H%M')+' Done...')
    # sys.exit()

end_time = dt.datetime.now()
print('===================')
print('Total elapsed time: {0}\n'.format(end_time-start_time))
