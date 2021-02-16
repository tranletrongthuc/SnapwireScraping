import json
from scraper import DetailPhotoScraper, CollectionScraper


def demo_get_detail_photos():
    image_ids = ['6009d98f3afa0b45826e98df', '5ecec4c4f21bfc5e7a711b7b', '600a435e908b853825cc0147']

    total_results = {}

    for img_id in image_ids:
        detail_photo_scraper = DetailPhotoScraper(img_id)
        detail_photo_scraper.get_deatil()

        total_results[img_id] = detail_photo_scraper.__dict__
        with open('result.json', 'w', encoding='utf-8') as json_file:
            json.dump(total_results, json_file)

def demo_get_collections():
    collection_scraper = CollectionScraper()
    collection_scraper.get_collection()

if __name__ == "__main__":
    demo_get_collections()