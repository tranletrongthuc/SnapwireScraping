import json
import os
from scraper import DetailPhotoScraper, CollectionScraper


def demo_get_detail_photos():
    image_ids = ['6009d98f3afa0b45826e98df', '5ecec4c4f21bfc5e7a711b7b', '600a435e908b853825cc0147']

    total_results = {}

    for img_id in image_ids:
        detail_photo_scraper = DetailPhotoScraper(img_id)
        detail_photo_scraper.get_detail()

        total_results[img_id] = detail_photo_scraper.__dict__
        with open('detail_photos.json', 'w', encoding='utf-8') as json_file:
            json.dump(total_results, json_file)

def demo_get_collections():
    collection_scraper = CollectionScraper()

    detail_collections_dir = os.path.join(collection_scraper.scrapped_data_dir,'detail_collections')
    if not os.path.exists(detail_collections_dir):
        os.mkdir(detail_collections_dir)

    detail_collections_path = os.path.join(detail_collections_dir,'collections.json')

    if not collection_scraper.check_existing_wrapped_file(detail_collections_path):
        collections = collection_scraper.get_collection()
        with open(detail_collections_path, 'w', encoding='utf-8') as json_file:
            json.dump(collections, json_file)
    else:
        with open(detail_collections_path, 'r', encoding='utf-8') as json_file:
            collections = json.load(json_file)

    collection_photos = {}
    detail_photos_dir = os.path.join(collection_scraper.scrapped_data_dir,'detail_photos')
    if not os.path.exists(detail_photos_dir):
        os.mkdir(detail_photos_dir)

    for collection in collections[:2] :
        collection_id, collection_url = collection['collection_id'], collection['collection_url']
        collection_dir = os.path.join(detail_photos_dir, collection_id)

        if not os.path.exists(collection_dir):
            os.mkdir(collection_dir)

        all_photos = collection_scraper.get_all_photos_of_collection(collection_url)
        collection_photos[collection_id] = all_photos
        print(f"{collection_id}: {len(all_photos)} photos found.")
        with open(os.path.join(collection_scraper.scrapped_data_dir,'all_photo.json'), 'w', encoding='utf-8') as json_file:
            json.dump(collection_photos, json_file)

        print(f"Start get all photos from {collection_id} collection")
        total_results = {}

        for photo_name in all_photos:
            detail_photo_scraper = DetailPhotoScraper(photo_name)

            saving_photo_path = os.path.join(collection_scraper.scrapped_data_dir,'detail_photos', collection_id, photo_name) + '.jpg'
            detail_photo_scraper.get_detail(saving_photo_path)

            total_results[photo_name] = detail_photo_scraper.__dict__
            with open(os.path.join(collection_scraper.scrapped_data_dir, f'{collection_id}.json'), 'w', encoding='utf-8') as json_file:
                json.dump(total_results, json_file)

if __name__ == "__main__":
    demo_get_collections()
    # demo_get_detail_photos()