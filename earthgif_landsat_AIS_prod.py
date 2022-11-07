from dask.distributed import LocalCluster,Client
import dask.utils
import dask.config
import stackstac
import dask.array as da
import pystac_client
import geopandas as gpd
import planetary_computer as pc
import xarray as xr

import shapely.geometry
import os

from PIL import Image, ImageDraw, ImageFont
from xarray.plot.utils import _rescale_imshow_rgb
import numpy as np
from typing import TYPE_CHECKING, BinaryIO, cast,Literal
import ntpath

#glacier = "thwaites"
#full_name= "Thwaites"

glacier = "amundsen"
full_name= "Amundsen bay"

def convert_bounds(bbox, invert_y=False):
    """
    Helper method for changing bounding box representation to leaflet notation
    ``(lon1, lat1, lon2, lat2) -> ((lat1, lon1), (lat2, lon2))``
    """
    x1, y1, x2, y2 = bbox
    if invert_y:
        y1, y2 = y2, y1
    return ((y1, x1), (y2, x2))

# define font size for Label
def fontSize(arr):
    img = Image.fromarray(arr)
    txt = "Hello World"
    fontsize = 1  # starting font size
    font = ImageFont.truetype("Calibri.ttf", fontsize)
    # portion of image width you want text width to be
    img_fraction = 0.10
    breakpoint = img_fraction * min(img.size[0],img.size[1])
    jumpsize = 75
    while True:
        if font.getlength(txt) < breakpoint:
            fontsize += jumpsize
        else:
            jumpsize = jumpsize // 2
            fontsize -= jumpsize
        font = ImageFont.truetype('Calibri.ttf', fontsize)
        if jumpsize <= 1:
            break

    return fontsize

def save_img(arr: xr.DataArray,
             timeStamp: bool = True,
             fontsize: int = 35,
             glacier:str = '.',
             date_position: Literal["ul", "ur", "ll", "lr"] = "ul",
             date_color: tuple[int, int, int] = (0, 0, 0),
             date_bg: tuple[int, int, int]  = (255, 255, 255)):
        
    date_img = str(arr.time.values)[0:10]
    # Rescale
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    data= _rescale_imshow_rgb(arr, vmin, vmax, True)

    # convert to U8
    u8 = (data * 255).astype("uint8").to_numpy()
    u8 = np.clip(u8, 0, 255, out=u8)
    u8 = np.moveaxis(u8, -3, -1) #??
    # Add alpha mask
    mask: np.ndarray = arr.isnull().data.any(axis=-3)
    alpha = (~mask).astype("uint8", copy=False) * 255
    frame = np.concatenate([u8, alpha[..., None]], axis=-1)

    #imgs = [Image.fromarray(frame) for frame in frames]
    img = Image.fromarray(frame)
    
     # Write timestamps onto each frame
       
    fnt =ImageFont.truetype("Calibri.ttf", fontsize)
    if timeStamp:
        
        label1 = full_name 
        label2 = 'Summer ' + date_img
        
        # get a drawing context
        d = ImageDraw.Draw(img)
        d = cast(ImageDraw.ImageDraw, d)

        width, height = img.size
        left, top, right, bottom = fnt.getbbox(label1)
        t_width = fnt.getlength(label1)
        t_height = abs(top - bottom)
        
        offset = max(0.03*width,0.03*height)
        if date_position[0] == "u":
            y = offset
        else:
            y = height - t_height - offset

        if date_position[1] == "l":
            x = offset
        else:
            x = width - t_width - offset

        border = t_height*0.1
        #if date_bg:
           # d.rectangle([(x-border, y-border),(x + t_width  , y + t_height)], fill=date_bg)
            #d.rectangle((x-border, y+2*t_height-border, x + t_width  + border, y+ 2*t_height + t_height+border), fill=date_bg)
        # draw text
        d.multiline_text((x, y), label1, font=fnt, fill=date_color)
        d.multiline_text((x, y+2*t_height), label2, font=fnt, fill=date_color)

    
    out_filename = os.path.join(os.getcwd(),glacier,ntpath.basename(filename).split('.')[0]+'_'+ date_img + '.png')
    
    img.save(
        out_filename,
        format="png"
    )
    print (glacier +': '+ date_img +' processed')

if __name__ == '__main__':
    cluster = LocalCluster(n_workers=10,
                        threads_per_worker=2,
                        dashboard_address=8787,
                        memory_limit='8GB')
    
    client = Client(cluster)
    print(client)


    #filename = "geojson file path"
    filename = glacier +".geojson"
    
    # read in AOI as a GeoDataFrame
    aoi = gpd.read_file(filename)
    bbox =aoi.unary_union.bounds

    # With the pystac_client module’s Client class, Open the STAC API. 
    datetimeRange=[]
    for t in range (2014,2022):
        datetimeRange.append(str(t)+"-01-01/"+str(t)+"-04-30")

    LandsatSTAC = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1')

    for dt in datetimeRange:    
        search = (
            LandsatSTAC
            .search(
                bbox=bbox,
                query =  {"eo:cloud_cover":{"lt":15}},
                datetime = dt, 
                collections = ["landsat-c2-l2"]
            )
        )
        
        items = pc.sign(search)
        
        print(dt +': ' +str(len(items))+ ' scenes found')

        stack = stackstac.stack(items,bounds_latlon=bbox, epsg = 32620, resolution=200)

        # use common_name for bands
        stack = stack.assign_coords(band=stack.common_name.fillna(stack.band).rename("band"))


        # keep rgb bands + Make annual median composites (`Q` means 2 quarters)
        composites = stack.sel(band=["red", "green", "blue"]).resample(time="Q").median("time")
        composites.ffill("time").bfill("time")

        ts = composites.persist()
        ts_local = ts.compute()

        # define the font size
        fontsize = fontSize(ts_local.isel(time=0)[0].to_numpy())
        for t in ts_local['time']:
            save_img(ts_local.sel(time=t), fontsize=fontsize, glacier=glacier,date_position="lr")
            

