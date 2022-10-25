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

if __name__ == '__main__':
    cluster = LocalCluster(n_workers=10,
                        threads_per_worker=2,
                        dashboard_address=8787,
                        memory_limit='6GB')
    
    client = Client(cluster)

    def convert_bounds(bbox, invert_y=False):
        """
        Helper method for changing bounding box representation to leaflet notation
        ``(lon1, lat1, lon2, lat2) -> ((lat1, lon1), (lat2, lon2))``
        """
        x1, y1, x2, y2 = bbox
        if invert_y:
            y1, y2 = y2, y1
        return ((y1, x1), (y2, x2))

    def save_img(arr: xr.DataArray,
                timeStamp: bool = True,
                date_position: Literal["ul", "ur", "ll", "lr"] = "ul",
                date_color: tuple[int, int, int] = (255, 255, 255),
                date_bg: tuple[int, int, int]  = (0, 0, 0)):
        
        date_img = str(arr.time.values)[0:4]
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
        fontsize = 35
        fnt =ImageFont.truetype("Calibri.ttf", fontsize)
        if timeStamp:
            
            label = 'Summer ' + date_img
            print (label)
            # get a drawing context
            d = ImageDraw.Draw(img)
            d = cast(ImageDraw.ImageDraw, d)

            width, height = img.size
            t_width, t_height = fnt.getsize(label)

            offset = 50
            if date_position[0] == "u":
                y = offset
            else:
                y = height - t_height - offset

            if date_position[1] == "l":
                x = offset
            else:
                x = width - t_width - offset

            border = 8
            if date_bg:
                d.rectangle((x-border, y-border, x + t_width  + border, y + t_height+border), fill=date_bg)
            # draw text
            d.multiline_text((x, y), label, font=fnt, fill=date_color)
        
        out_filename = './petermann/'+ ntpath.basename(filename).split('.')[0] + '_' + date_img + '.png'
        
        img.save(
            out_filename,
            format="png"
        )
        print(out_filename + ' saved')  

    #filename = "geojson file path"
    filename = "petermann.geojson"
    
    # read in AOI as a GeoDataFrame
    aoi = gpd.read_file(filename)
    bbox =aoi.unary_union.bounds

    # With the pystac_client module’s Client class, Open the STAC API. 
    datetimeRange=[]
    for t in range (2014,2022):
        datetimeRange.append(str(t)+"-07-01/"+str(t)+"-10-15")

    LandsatSTAC = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1')


    for dt in datetimeRange:    
        search = (
            LandsatSTAC
            .search(
                bbox=bbox,
                query =  {"eo:cloud_cover":{"lt":1}},
                datetime = dt, 
                collections = ["landsat-c2-l2"]
                    
            )
        )
        
        items = pc.sign(search)
        
        print(dt +': ' +str(len(items))+ ' scenes found')

        stack = stackstac.stack(items,bounds_latlon=bbox, epsg = 32620, resolution=50)

        # use common_name for bands
        stack = stack.assign_coords(band=stack.common_name.fillna(stack.band).rename("band"))


        # keep rgb bands + Make annual median composites (`Q` means 2 quarters)
        composites = stack.sel(band=["red", "green", "blue"]).resample(time="A").median("time")
        composites.ffill("time").bfill("time")

        ts = composites.persist()
        ts_local = ts.compute()

        for t in ts_local['time']:
            save_img(ts_local.sel(time=t))


