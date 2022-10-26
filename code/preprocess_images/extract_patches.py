OPENSLIDE_PATH = r'C:\\Users\\dantg\\openslide-win64-20220811\\openslide-win64-20220811\\bin'

import os
import matplotlib.pyplot as plt

if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
        from openslide.deepzoom import DeepZoomGenerator
else:
    import openslide
    from openslide.deepzoom import DeepZoomGenerator


path = '../../../WSS1-v1/train/'
slideName = '01.svs'
anno = '01_anno.json'
slide = openslide.OpenSlide(f'{path}/{slideName}')
width, height = slide.dimensions
print('Slide height: ', height)
print('Slide width: ', width)

tileSize = 224
tiles=DeepZoomGenerator(slide, tile_size=tileSize, overlap=0, limit_bounds=False)
print(tiles.tile_count)
print(tiles.level_count)

level = len(tiles.level_tiles)-1
col, rows= tiles.level_tiles[level]
print('level tiles: ', tiles.level_tiles[level])
print('num rows: ', rows)
print('num cols: ', col)
# tile1=tiles.get_tile(level,(90, 70))
# plt.figure(figsize=(5,5))
# plt.imshow(tile1)
# plt.show()
coords = tiles.get_tile_coordinates(level,(40, 70))
print('coordinates: ', coords)
for c in range(col):
    for r in range(rows):
        tile1=tiles.get_tile(level,(c, r))
        coords = tiles.get_tile_coordinates(level,(c, r))

        ## Check for each (x,y) which class it belongs to
        # print('coordinates: ', coords)
        plt.figure(figsize=(5,5))
        plt.imshow(tile1)
        plt.show()
        break
        
    break