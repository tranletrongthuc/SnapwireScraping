import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import time
import os
import urllib.request

class CollectionScraper:
    def __init__(self):
        self.collections_url = 'https://community.snapwire.co/collections'
        self.collections = []
        self.browser_path = './operadriver.exe'
        self.browser = None
        self.scrapped_data_dir = './scrapped_data'
        self.init_browser()

    def init_browser(self):
        self.browser = webdriver.Chrome(executable_path=self.browser_path)

    def find_element_by_class_name_until_end(self, class_name, loop_time = 5, limit_result = None):
        result = []
        check_height = self.browser.execute_script("return document.body.scrollHeight;")
        while (True):
            self.browser.execute_script(f"window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(loop_time)
            new_height = self.browser.execute_script("return document.body.scrollHeight;")
            # print(f"Check height: {check_height} - new height {new_height}")
            if check_height != new_height:
                check_height = new_height
                result = self.browser.find_elements_by_class_name(class_name)
                # print(f"{class_name} - {loop_time} - {len(result)} photos found.")
            else:
                break

        return result

    def check_existing_wrapped_file(self, file_name):
        return os.path.exists(os.path.join(self.scrapped_data_dir, file_name))

    def get_collection(self):
        self.browser.get(self.collections_url)
        # self.collections = self.find_element_by_class_name_until_end('request-title')
        # self.collections = self.find_element_by_class_name_until_end('sw-card sw-card-request sw-card-background collection box')
        self.collections = self.find_element_by_class_name_until_end('grid-item')

        self.collections = [{   'collection_name': collection.find_elements_by_class_name('request-title')[0].text,
                                'collection_url': collection.find_elements_by_tag_name('a')[0].get_attribute('href'),
                                'collection_id': collection.find_elements_by_class_name('request-title')[0].text.replace(" ","-").lower(),
                             } for collection in self.collections]
        print(self.collections)
        return self.collections

    def get_all_photos_of_collection(self, collection_url):
        self.browser.get(collection_url)
        photo_wrappers = self.find_element_by_class_name_until_end('photo-wrapper', 5, 500)
        all_photos = []
        for wrapper in photo_wrappers:
            photo_url = wrapper.find_elements_by_tag_name('a')[0].get_attribute('href')
            all_photos.append(photo_url.split('/')[-1])

        return all_photos

class DetailPhotoScraper:

    def __init__(self, photo_id):
        self.photo_id = photo_id
        self.page_url = "https://community.snapwire.co/photo/detail/" + self.photo_id
        self.like_count = 0,
        self.photographer = ''
        self.keyworks = []
        self.caption = ''
        self.photo_url = ''
        self.snapwire_detail = {}

    def get_detail(self, saving_path):

        page_content = requests.get(self.page_url).text

        soup = BeautifulSoup(page_content, "lxml")

        # get Like-count
        like_count_soup = soup.find('h4', {"class": "light like-count"})
        if like_count_soup is not None:
            try:
                self.like_count = int(like_count_soup.text)
            except:
                pass

        # get Photographer info
        photographer_soup = soup.find('div','photographer-info').find('h4')
        if photographer_soup is not None:
            self.photographer = photographer_soup.text

        detail_info_soup = soup.find("section", {"class": "detail-info"})
        if detail_info_soup is not None:
            # get Keywords and caption
            keywork_area_soup = [area for area in detail_info_soup.findAll('div', {"class": "col-md-8"}) if (area.find('h5') is not None and area.find('h5').text.lower().__contains__('keywords'))]
            if len(keywork_area_soup) > 0:
                self.keyworks = [keywork_soup.text for keywork_soup in keywork_area_soup[0].find_all("a")]

            caption = detail_info_soup.find('p', {'id':'caption-text'})
            if caption is not None:
                caption = caption.text
                caption = caption.encode('ascii', 'ignore').decode('utf-8')
                self.caption = caption.strip()

        # get Photo URL:
        photo_url_soup = soup.find('img', {"class": "margin-bottom"})
        if photographer_soup is not None:
            self.photo_url = photo_url_soup.attrs['src']
            urllib.request.urlretrieve(self.photo_url, saving_path)

        # get Photo detail

