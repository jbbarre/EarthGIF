
import dask.distributed
import pystac_client
import stackstac
import dask.array as da
from geogif import gif, dgif
import geopandas as gpd
import planetary_computer as pc

#Load STAC items into xarray Datasets. Process locally or distribute data loading and computation with Dask.
from odc.stac import configure_rio, load
from IPython.display import HTML, display
import folium
import folium.plugins
from branca.element import Figure
import shapely.geometry


client = dask.distributed.Client()
display(client)


def convert_bounds(bbox, invert_y=False):
    """
    Helper method for changing bounding box representation to leaflet notation
    ``(lon1, lat1, lon2, lat2) -> ((lat1, lon1), (lat2, lon2))``
    """
    x1, y1, x2, y2 = bbox
    if invert_y:
        y1, y2 = y2, y1
    return ((y1, x1), (y2, x2))


#petermann - or create one with https://geojson.io/#map=2/20.0/0.0 and follow https://aws.amazon.com/fr/blogs/apn/transforming-geospatial-data-to-cloud-native-frameworks-with-element-84-on-aws/

#filename = "geojson file path"# read in AOI as a GeoDataFrame
filename = "petermann.geojson"# read in AOI as a GeoDataFrame
# read in AOI as a GeoDataFrame
aoi = gpd.read_file(filename)


%time

bbox =aoi.unary_union.bounds

LandsatSTAC = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1')

search = (
    LandsatSTAC
    .search(
        bbox=bbox,
        query =  {"eo:cloud_cover":{"lt":15}},
        collections = ["landsat-8-c2-l2"],
        datetime = "2020-01-01/2020-12-30"        
    )
)

%time
items = pc.sign(search)
len(items)

stack = stackstac.stack(items, bounds_latlon=bbox, epsg = 32620, resolution=30)

#scl = stack.sel(band=["SCL"])
# Sentinel-2 Scene Classification Map: nodata, saturated/defective, dark, cloud shadow, cloud med. prob., cloud high prob., cirrus
#invalid = da.isin(scl, [0, 1, 2, 3, 8, 9, 10])
#valid = stack.where(~invalid)

rgb = stack.sel(band=["SR_B4", "SR_B3", "SR_B2"])
quarterly = rgb.resample(time="M").median()

ts = quarterly.persist()
cleaned = ts.ffill("time").bfill("time")

gif_data = dgif(cleaned, bytes=True).compute()
with open("petermann_planetary.gif", "wb") as f:
    f.write(gif_data)



