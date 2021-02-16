import requests
from bs4 import BeautifulSoup
import json




class DetailPhotoScraper:

    def __init__(self, image_id):
        self.image_id = image_id
        self.page_url = "https://community.snapwire.co/photo/detail/" + self.image_id
        self.page_content = ""

    def get_page_content(self):
        self.page_content = requests.get(self.page_url).text

    def parse(self):
        result = {
            'like-count' : 0,
            'photographer' : '',
            'keyworks' : [],
            'caption' : [],
            'photo_url': '',
            'snapwire_detail' : {}
        }

        soup = BeautifulSoup(self.page_content, "lxml")

        # get Like-count
        like_count_soup = soup.find('h4', {"class": "light like-count"})
        like_count = int(like_count_soup.text)

        # get Photographer info
        photographer_soup = soup.find('div','photographer-info').find('h4')
        photographer = photographer_soup.text

        detail_info_soup = soup.find("section", {"class": "detail-info"})
        # get Keywords and caption
        keywork_area_soup = [area for area in detail_info_soup.findAll('div', {"class": "col-md-8"}) if (area.find('h5') is not None and area.find('h5').text.lower().__contains__('keywords'))]
        keyworks = []
        if len(keywork_area_soup) > 0:
            keyworks = [keywork_soup.text for keywork_soup in keywork_area_soup[0].find_all("a")]

        caption = detail_info_soup.find('p', {'id':'caption-text'}).text
        caption = caption.encode('ascii', 'ignore').decode('utf-8')
        caption = caption.strip()

        # get Photo URL:
        photo_url_soup = soup.find('img', {"class": "margin-bottom"})
        photo_url = photo_url_soup.attrs['src']
        # get Photo detail

        result['like-count'] = like_count
        result['photographer'] = photographer
        result['keyworks'] = keyworks
        result['caption'] = caption
        result['photo_url'] = photo_url

        print(f"Finsih scraping {self.image_id}")
        return result




if __name__ == "__main__":
    image_ids = ['6009d98f3afa0b45826e98df', '5ecec4c4f21bfc5e7a711b7b', '600a435e908b853825cc0147']

    total_results = {}

    for img_id in image_ids:
        d = DetailPhotoScraper(img_id)
        d.get_page_content()
        result = d.parse()

        total_results[img_id] = result
        with open('result.json', 'w', encoding='utf-8') as json_file:
            json.dump(total_results, json_file)

    with open('result.json', 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        print(data)