
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:32:20 2017

@author: avanetten

copied from https://github.com/CosmiQ/apls/tree/master/src
"""

from __future__ import print_function

import osmnx as ox
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import geopandas as gpd
from osgeo import gdal, ogr, osr
import cv2
import subprocess
import shapely
from shapely.geometry import MultiLineString
from matplotlib.patches import PathPatch
import matplotlib.path

###############################################################################
### Previsously in apls.py
###############################################################################
def plot_metric(C, diffs, routes_str=[], 
                figsize=(10,5), scatter_png='', hist_png='',
                scatter_alpha=0.3, scatter_size=2, scatter_cmap='jet', dpi=300):
    ''' Plot outpute of cost metric in both scatterplot and histogram format'''
    
    # plot diffs
    title = 'Path Length Similarity: ' + str(np.round(C,2)) 
    fig, (ax0) = plt.subplots(1, 1, figsize=(1*figsize[0], figsize[1]))
    #ax0.plot(diffs)
    ax0.scatter(range(len(diffs)), diffs, s=scatter_size, c=diffs, 
                alpha=scatter_alpha, 
                cmap=scatter_cmap)
    if len(routes_str) > 0:
        xticklabel_pad = 0.1
        ax0.set_xticks(range(len(diffs)))
        ax0.set_xticklabels(routes_str, rotation=50, fontsize=4)
        ax0.tick_params(axis='x', which='major', pad=xticklabel_pad)

    ax0.set_ylabel('Length Diff (Normalized)')
    ax0.set_xlabel('Path ID')
    ax0.set_title(title)
    #plt.tight_layout()
    if scatter_png:
        plt.savefig(scatter_png, dpi=dpi)
    
    # plot and plot diffs histo
    bins = np.linspace(0, 1, 30)
    bin_centers = np.mean( zip(bins, bins[1:]), axis=1)
    #digitized = np.digitize(diffs, bins)
    #bin_means = [np.array(diffs)[digitized == i].mean() for i in range(1, len(bins))]
    hist, bin_edges = np.histogram(diffs, bins=bins)
    fig, ax1 = plt.subplots(nrows=1, ncols=1,  figsize=figsize)
    #ax1.plot(bins[1:],hist, type='bar')
    #ax1.bar(bin_centers, hist, width=bin_centers[1]-bin_centers[0] )
    ax1.bar(bin_centers, 1.*hist/len(diffs), width=bin_centers[1]-bin_centers[0] )
    ax1.set_xlim([0,1])
    #ax1.set_ylabel('Num Routes')
    ax1.set_ylabel('Frac Num Routes')
    ax1.set_xlabel('Length Diff (Normalized)')
    ax1.set_title('Length Diff Histogram - Score: ' + str(np.round(C,2)) )
    ax1.grid(True)
    #plt.tight_layout()
    if hist_png:
        plt.savefig(hist_png, dpi=dpi)
    
    return

###############################################################################
###############################################################################
# For plotting the buffer...
#https://sgillies.net/2010/04/06/painting-punctured-polygons-with-matplotlib.html
def ring_coding(ob):
    # The codes will be all "LINETO" commands, except for "MOVETO"s at the
    # beginning of each subpath
    n = len(ob.coords)
    codes = np.ones(n, dtype=matplotlib.path.Path.code_type) * matplotlib.path.Path.LINETO
    codes[0] = matplotlib.path.Path.MOVETO
    return codes

#https://sgillies.net/2010/04/06/painting-punctured-polygons-with-matplotlib.html
def pathify(polygon):
    # Convert coordinates to path vertices. Objects produced by Shapely's
    # analytic methods have the proper coordinate order, no need to sort.
    vertices = np.concatenate(
                    [np.asarray(polygon.exterior)]
                    + [np.asarray(r) for r in polygon.interiors])
    codes = np.concatenate(
                [ring_coding(polygon.exterior)]
                + [ring_coding(r) for r in polygon.interiors])
    return matplotlib.path.Path(vertices, codes)
###############################################################################

###############################################################################
def plot_buff(G_, ax, buff=20, color='yellow', alpha=0.3, 
              title='Proposal Snapping',
              title_fontsize=8, outfile='', 
              dpi=200,
              verbose=False):
    '''plot buffer around graph using shapely buffer'''
        
    # get lines
    line_list = []    
    for u, v, key, data in G_.edges(keys=True, data=True):
        if verbose:
            print ("u, v, key:", u, v, key) 
            print ("  data:", data)
        geom = data['geometry']
        line_list.append(geom)
    
    mls = MultiLineString(line_list)
    mls_buff = mls.buffer(buff)

    if verbose: 
        print ("type(mls_buff) == MultiPolygon:", 
               type(mls_buff) == shapely.geometry.MultiPolygon)
    
    if type(mls_buff) == shapely.geometry.Polygon:
        mls_buff_list = [mls_buff]
    else:
        mls_buff_list = mls_buff
    
    for poly in mls_buff_list:
        x,y = poly.exterior.xy
        coords = np.stack((x,y), axis=1)
        interiors = poly.interiors
        #coords_inner = np.stack((x_inner,y_inner), axis=1)
        
        if len(interiors) == 0:
            #ax.plot(x, y, color='#6699cc', alpha=0.0, linewidth=3, solid_capstyle='round', zorder=2)
            ax.add_patch(matplotlib.patches.Polygon(coords, alpha=alpha, color=color))
        else:
            path = pathify(poly)
            patch = PathPatch(path, facecolor=color, edgecolor=color, alpha=alpha)
            ax.add_patch(patch)
 
    ax.axis('off')
    if len(title) > 0:
        ax.set_title(title, fontsize=title_fontsize)
    #outfile = os.path.join(res_dir, 'gt_raw_buffer.png')
    if outfile:
        plt.savefig(outfile, dpi=dpi)
    return ax

###############################################################################
def plot_node_ids(G, ax, node_list=[], alpha=0.8, fontsize=8, plot_node=False, node_size=15,
                  node_color='orange'):
    '''
    for label, x, y in zip(labels, data[:, 0], data[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    '''
    Gnodes = set(G.nodes())

    if len(node_list) == 0:
        nodes = G.nodes()
    else:
        nodes = node_list
    for n in nodes: #G.nodes():
        if n not in Gnodes:
            continue
        x,y = G.node[n]['x'], G.node[n]['y']
        if plot_node:
            ax.scatter(x,y, s=node_size, color=node_color)
        ax.annotate(str(n), xy=(x, y), alpha=alpha, fontsize=fontsize)
        
    return ax

###############################################################################
def get_graph_extent(G_):
    '''min and max x and y'''
    xall = [G_.node[n]['x'] for n in G_.nodes()]
    yall = [G_.node[n]['y'] for n in G_.nodes()]
    xmin, xmax = np.min(xall), np.max(xall)
    ymin, ymax = np.min(yall), np.max(yall)
    dx, dy = xmax-xmin, ymax-ymin   
    return xmin, xmax, ymin, ymax, dx, dy
###############################################################################
###############################################################################

### Conversion and data formatting functions
###############################################################################
def convert_to_8Bit(inputRaster, outputRaster,
                           outputPixType='Byte',
                           outputFormat='GTiff',
                           rescale_type='rescale',
                           percentiles=[2, 98]):
    '''
    Convert 16bit image to 8bit
    rescale_type = [clip, rescale]
        if clip, scaling is done sctricly between 0 65535 
        if rescale, each band is rescaled to a min and max 
        set by percentiles
    '''

    srcRaster = gdal.Open(inputRaster)
    cmd = ['gdal_translate', '-ot', outputPixType, '-of', 
           outputFormat]
    
    # iterate through bands
    for bandId in range(srcRaster.RasterCount):
        bandId = bandId+1
        band = srcRaster.GetRasterBand(bandId)
        if rescale_type == 'rescale':
            bmin = band.GetMinimum()        
            bmax = band.GetMaximum()
            # if not exist minimum and maximum values
            if bmin is None or bmax is None:
                (bmin, bmax) = band.ComputeRasterMinMax(1)
            # else, rescale
            band_arr_tmp = band.ReadAsArray()
            bmin = np.percentile(band_arr_tmp.flatten(), 
                                 percentiles[0])
            bmax= np.percentile(band_arr_tmp.flatten(), 
                                percentiles[1])

        else:
            bmin, bmax = 0, 65535

        cmd.append('-scale_{}'.format(bandId))
        cmd.append('{}'.format(bmin))
        cmd.append('{}'.format(bmax))
        cmd.append('{}'.format(0))
        cmd.append('{}'.format(255))

    cmd.append(inputRaster)
    cmd.append(outputRaster)
    print ("Conversion command:", cmd)
    subprocess.call(cmd)
    
    return

###############################################################################
def load_multiband_im(image_loc, method='gdal'):
    '''
    Use gdal to laod multiband files.  If image is 1-band or 3-band and 8bit, 
    cv2 will be much faster, so set method='cv2'
    Return numpy array
    '''
    
    im_gdal = gdal.Open(image_loc)
    nbands = im_gdal.RasterCount
    
    # use gdal, necessary for 16 bit
    if method == 'gdal':
        bandlist = []
        for band in range(1, nbands+1):
            srcband = im_gdal.GetRasterBand(band)
            band_arr_tmp = srcband.ReadAsArray()
            bandlist.append(band_arr_tmp)
        img = np.stack(bandlist, axis=2)

    # use cv2, which is much faster if data is 8bit and 1-band or 3-band
    elif method == 'cv2':
        # check data type (must be 8bit)
        srcband = im_gdal.GetRasterBand(1)
        band_arr_tmp = srcband.ReadAsArray()
        if band_arr_tmp.dtype == 'uint16': 
            print ("cv2 cannot open 16 bit images")
            return []
        # ingest
        if nbands == 1:
            img = cv2.imread(image_loc, 0)
        elif nbands == 3:
            img = cv2.imread(image_loc, 1)
        else:
            print ("cv2 cannot open images with", nbands, "bands")
            return []

    return img


###############################################################################
def latlon2pixel(lat, lon, input_raster='', targetsr='', geom_transform=''):
    '''
    Convert latitude, longitude coords to pixexl coords.
    From spacenet geotools
    '''

    sourcesr = osr.SpatialReference()
    sourcesr.ImportFromEPSG(4326)

    geom = ogr.Geometry(ogr.wkbPoint)
    geom.AddPoint(lon, lat)

    if targetsr == '':
        src_raster = gdal.Open(input_raster)
        targetsr = osr.SpatialReference()
        targetsr.ImportFromWkt(src_raster.GetProjectionRef())
    coord_trans = osr.CoordinateTransformation(sourcesr, targetsr)
    if geom_transform == '':
        src_raster = gdal.Open(input_raster)
        transform = src_raster.GetGeoTransform()
    else:
        transform = geom_transform

    x_origin = transform[0]
    # print(x_origin)
    y_origin = transform[3]
    # print(y_origin)
    pixel_width = transform[1]
    # print(pixel_width)
    pixel_height = transform[5]
    # print(pixel_height)
    geom.Transform(coord_trans)
    # print(geom.GetPoint())
    x_pix = (geom.GetPoint()[0] - x_origin) / pixel_width
    y_pix = (geom.GetPoint()[1] - y_origin) / pixel_height

    return (x_pix, y_pix)

###############################################################################
def create_buffer_geopandas(geoJsonFileName, bufferDistanceMeters=2, 
                          bufferRoundness=1, projectToUTM=True):
    '''
    Create a buffer around the lines of the geojson. 
    Return a geodataframe.
    '''
    
    try:
        inGDF = gpd.read_file(geoJsonFileName)
    except:
        return []
    
    # set a few columns that we will need later
    inGDF['type'] = inGDF['road_type'].values            
    inGDF['class'] = 'highway'  
    inGDF['highway'] = 'highway'  
    
    if len(inGDF) == 0:
        return []

    # Transform gdf Roadlines into UTM so that Buffer makes sense
    if projectToUTM:
        tmpGDF = ox.project_gdf(inGDF)
    else:
        tmpGDF = inGDF

    gdf_utm_buffer = tmpGDF

    # perform Buffer to produce polygons from Line Segments
    gdf_utm_buffer['geometry'] = tmpGDF.buffer(bufferDistanceMeters,
                                                bufferRoundness)

    gdf_utm_dissolve = gdf_utm_buffer.dissolve(by='class')
    gdf_utm_dissolve.crs = gdf_utm_buffer.crs

    if projectToUTM:
        gdf_buffer = gdf_utm_dissolve.to_crs(inGDF.crs)
    else:
        gdf_buffer = gdf_utm_dissolve

    return gdf_buffer


###############################################################################
def gdf_to_array(gdf, im_file, output_raster, burnValue=150):
    
    '''
    Turn geodataframe to array, save as image file with non-null pixels 
    set to burnValue
    '''

    NoData_value = 0      # -9999

    gdata = gdal.Open(im_file)
    
    # set target info
    target_ds = gdal.GetDriverByName('GTiff').Create(output_raster, 
                                                     gdata.RasterXSize, 
                                                     gdata.RasterYSize, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(gdata.GetGeoTransform())
    
    # set raster info
    raster_srs = osr.SpatialReference()