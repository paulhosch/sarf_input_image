import requests
import zipfile
import io
from pathlib import Path
import geopandas as gpd
import math

def load_token(token_file: Path) -> str:
    """Load Zenodo API token from file."""
    #zenodo api token: nLOGIrtBhyrJEo4kECFwHs7I4vzWRnNQIznfIIya0iLjecZxP6m3HeVDYNYz

    return token_file.read_text().strip()


def parse_corner(corner: str) -> tuple[int, int]:
    """Parse a tile or zip corner string like 'n00e097' into (lat, lon)."""
    lat_sign = 1 if corner[0] == 'n' else -1
    lon_sign = 1 if corner[3] == 'e' else -1
    lat = lat_sign * int(corner[1:3])
    lon = lon_sign * int(corner[4:7])
    return lat, lon


def format_tile_name(lat: int, lon: int) -> str:
    """Format integer lat/lon to tile name 'n00e097.tif'."""
    lat_prefix = 'n' if lat >= 0 else 's'
    lon_prefix = 'e' if lon >= 0 else 'w'
    return f"{lat_prefix}{abs(lat):02d}{lon_prefix}{abs(lon):03d}.tif"


def compute_needed_tiles(aoi_path: Path) -> list[str]:
    """
    Given an AOI shapefile, compute list of 1x1° tile filenames needed to cover it.
    """
    gdf = gpd.read_file(aoi_path)
    # Ensure AOI in lat/lon
    gdf = gdf.to_crs(epsg=4326)
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    min_lon, min_lat, max_lon, max_lat = bounds

    # Determine integer tile indices
    lat_start = math.floor(min_lat)
    lat_end = math.floor(max_lat)
    lon_start = math.floor(min_lon)
    lon_end = math.floor(max_lon)

    tiles = []
    for lat in range(lat_start, lat_end + 1):
        for lon in range(lon_start, lon_end + 1):
            tiles.append(format_tile_name(lat, lon))
    return tiles


def assign_tiles_to_zips(tile_names: list[str]) -> dict[str, list[str]]:
    """
    Assign each tile to its covering 30x30° zip archive.
    Returns mapping zip_basename -> list of tile names.
    """
    mapping: dict[str, list[str]] = {}
    for tile in tile_names:
        lat, lon = parse_corner(tile[:-4])
        # compute SW corner of zip (multiples of 30)
        sw_lat = (math.floor(lat / 30) * 30)
        sw_lon = (math.floor(lon / 30) * 30)
        ne_lat = sw_lat + 30
        ne_lon = sw_lon + 30
        # format corners
        sw = f"{('n' if sw_lat>=0 else 's')}{abs(sw_lat):02d}{('e' if sw_lon>=0 else 'w')}{abs(sw_lon):03d}"
        ne = f"{('n' if ne_lat>=0 else 's')}{abs(ne_lat):02d}{('e' if ne_lon>=0 else 'w')}{abs(ne_lon):03d}"
        zip_base = f"{sw}-{ne}_FathomDEM_v1-0.zip"
        mapping.setdefault(zip_base, []).append(tile)
    return mapping


def fetch_record_files(record_id: int, token: str) -> list[dict]:
    """Fetch file list from Zenodo record via API."""
    headers = {'Authorization': f'Bearer {token}'}
    url = f'https://zenodo.org/api/records/{record_id}'
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return r.json()['files']


def download_and_extract(zips: dict[str, list[str]], record_files: list[dict], token: str, out_dir: Path):
    """
    For each zip, download and extract only the requested tiles.
    """
    headers = {'Authorization': f'Bearer {token}'}
    out_dir.mkdir(parents=True, exist_ok=True)
    for zip_name, tiles in zips.items():
        # find zip URL
        file_info = next((f for f in record_files if f['key'] == zip_name), None)
        if not file_info:
            print(f"Warning: {zip_name} not found in record.")
            continue
        print(f"Downloading {zip_name}...")
        resp = requests.get(file_info['links']['self'], headers=headers)
        resp.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        for tile in tiles:
            if tile in z.namelist():
                print(f"Extracting {tile}...")
                with z.open(tile) as src, open(out_dir / tile, 'wb') as dst:
                    dst.write(src.read())
            else:
                print(f"Tile {tile} not in {zip_name}.")
        z.close()


def main(aoi_path: str, token_file: str, output_folder: str = 'tiles'):
    token = load_token(Path(token_file))
    tiles = compute_needed_tiles(Path(aoi_path))
    zips = assign_tiles_to_zips(tiles)
    record_files = fetch_record_files(14511570, token)
    download_and_extract(zips, record_files, token, Path(output_folder))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Download FathomDEM tiles covering an AOI')
    parser.add_argument('aoi_shapefile', help='Path to AOI shapefile')
    parser.add_argument('token_file', help='Path to Zenodo token file')
    parser.add_argument('--output', default='tiles', help='Output directory')
    args = parser.parse_args()
    main(args.aoi_shapefile, args.token_file, args.output)
