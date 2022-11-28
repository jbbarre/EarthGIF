
import dask.dataframe as dd
from dask.distributed import Client


if __name__ == '__main__':
    client = Client(processes=False)
    print(client)
    ddf = dd.read_parquet(
        "s3://dask-data/nyc-taxi/nyc-2015.parquet/part.*.parquet",
        columns=["passenger_count", "tip_amount"],
        storage_options={"anon": True},
    )

    result = ddf.groupby("passenger_count").tip_amount.mean().compute()
    print(result)
