import json
import os
import conf
from scraper import DetailPhotoScraper, CollectionScraper

def init_app():
    if not os.path.exists(conf.scrapped_data_dir):
        os.mkdir(conf.scrapped_data_dir)
    if not os.path.exists(conf.detail_collections_dir):
        os.mkdir(conf.detail_collections_dir)
    if not os.path.exists(conf.detail_photos_dir):
        os.mkdir(conf.detail_photos_dir)

def get_collection_details():
    collection_scraper = CollectionScraper()
    collection_scraper.get_collection()

def get_photo_details():

    with open(os.path.join(conf.scrapped_data_dir, 'collection_info.json'), 'r', encoding='utf-8') as json_file:
        collection_data = json.load(json_file)

    for collection in collection_data:
        total_results = {}

        print(f"Start getting all photos from \"{collection['id']}\" collection. {len(collection['photos'])} photos found.")

        if not os.path.exists(os.path.join(conf.detail_photos_dir, collection['id'])):
            os.mkdir(os.path.join(conf.detail_photos_dir, collection['id']))

        for photo_name in collection['photos']:
            detail_photo_scraper = DetailPhotoScraper(photo_name)

            saving_photo_path = os.path.join(conf.detail_photos_dir, collection['id'], photo_name) + '.jpg'
            detail_photo_scraper.get_detail(saving_photo_path)

            total_results[photo_name] = detail_photo_scraper.__dict__
        with open(os.path.join(conf.detail_collections_dir, f'{collection["id"]}.json'), 'w',
                  encoding='utf-8') as json_file_2:
            json.dump(total_results, json_file_2)

def main():
    init_app()
    get_photo_details()



if __name__ == "__main__":
    main()
