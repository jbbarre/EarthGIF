import dask.distributed 
import pystac_client
import stackstac
from geogif import gif, dgif
import dask.array as da
import geopandas as gpd
import numpy as np

if __name__ == '__main__':
    client = dask.distributed.Client()
    print(client)

    filename = "petermann.geojson"# read in AOI as a GeoDataFrame
    # read in AOI as a GeoDataFrame
    aoi = gpd.read_file(filename)

    bbox =aoi.unary_union.bounds

    items = (
        pystac_client.Client.open("https://earth-search.aws.element84.com/v0")
        .search(
            bbox=bbox,
            collections=["sentinel-s2-l2a-cogs"],
            query={"eo:cloud_cover":{"lt":1},"sentinel:valid_cloud_cover": {"eq": True}},
            datetime = "2021-01-01/2021-12-31"        
        )
    )

    print(f'{items.matched()} scenes founded')

    stack = stackstac.stack(items.item_collection(), bounds_latlon=bbox, epsg = 32620, resolution=30)

    # Then mask out bad (cloudy) pixels, according to the Sentinel-2 SCL Scene Classification Map, and take the temporal median of each quarter (three months) to hopefully get an okay-looking cloud-free frame representative of those three months.

    scl = stack.sel(band=["SCL"])
    # Sentinel-2 Scene Classification Map: nodata, saturated/defective, dark, cloud shadow, cloud med. prob., cloud high prob., cirrus
    invalid = da.isin(scl, [0, 1, 2, 3, 8, 9, 10])
    valid = stack.where(~invalid)

    rgb = valid.sel(band=["B04", "B03", "B02"])

    quarterly = rgb.resample(time="W").median()
    quarterly

    ts = quarterly.persist()
    cleaned = ts.ffill("time").bfill("time")

    gif_bytes = dgif(cleaned, bytes=True).compute()

    with open("petermann_dask.gif", "wb") as f:
        f.write(gif_bytes)
