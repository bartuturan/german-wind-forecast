import numpy as np

from windfc.config import BBox
from windfc.decode import crop_index, grid_coords


def test_grid_coords_north_to_south_and_wrapped():
    # global 0.25 deg grid as ECMWF open data writes it: 90 -> -90, lon starting at 180
    lats, lons = grid_coords(1440, 721, 90.0, 180.0, 0.25, 0.25, j_scans_positively=False)
    assert lats[0] == 90.0 and lats[-1] == -90.0
    assert lons.min() == -180.0 and lons.max() == 179.75


def test_crop_orders_ascending_and_selects_values():
    lats, lons = grid_coords(8, 5, 2.0, 358.0, 1.0, 1.0, j_scans_positively=False)
    # lats 2..-2 (north to south), lons 358,359,0..5 -> -2,-1,0..5
    field = np.add.outer(lats * 100, lons)  # value encodes its own lat/lon
    rows, cols = crop_index(lats, lons, BBox(lat_min=-1, lat_max=1, lon_min=-1, lon_max=2))
    assert lats[rows].tolist() == [-1, 0, 1]
    assert lons[cols].tolist() == [-1, 0, 1, 2]
    sub = field[np.ix_(rows, cols)]
    assert sub[0, 0] == -1 * 100 + -1
    assert sub[-1, -1] == 1 * 100 + 2
