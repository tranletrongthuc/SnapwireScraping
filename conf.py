from pathlib import Path
import os

base_dir = Path(__file__).parent
scrapped_data_dir = os.path.join(base_dir, 'scrapped_data')
detail_collections_dir = os.path.join(scrapped_data_dir, 'detail_collections')
detail_photos_dir = os.path.join(scrapped_data_dir, 'detail_photos')
collection_info_path = os.path.join(scrapped_data_dir, 'collection_info.json')
browser_path = os.path.join(base_dir,'operadriver.exe')
