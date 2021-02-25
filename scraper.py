import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import time
import os
import json
import sys
import conf
import urllib.request
from fake_useragent import UserAgent

class CollectionScraper:
    def __init__(self):
        self.collections_url = 'https://community.snapwire.co/collections/'
        self.collections = []
        self.browser = self.init_browser()

    def init_browser(self):
        browser = webdriver.Opera(executable_path=conf.browser_path)
        return browser

    def find_element_by_class_name_until_end(self, browser, class_name, loop_time = 5):
        result = []
        check_height = browser.execute_script("return document.body.scrollHeight;")
        while (True):
            browser.execute_script(f"window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(loop_time)
            new_height = browser.execute_script("return document.body.scrollHeight;")
            # print(f"Check height: {check_height} - new height {new_height}")
            if check_height != new_height:
                check_height = new_height
                result = browser.find_elements_by_class_name(class_name)
                # print(f"{class_name} - {loop_time} - {len(result)} photos found.")
            else:
                break

        return result

    def check_existing_wrapped_file(self, detail_collections_path):
        return os.path.exists(detail_collections_path)

    def get_collection(self):
        print("Start getting collections from Snapwire.")

        collection_infos = []

        self.browser.get(self.collections_url)
        self.collections = self.find_element_by_class_name_until_end(self.browser, 'grid-item')
        for collection in self.collections:
            collection_info = {}
            collection_info['name'] = collection.find_elements_by_class_name('request-title')[0].text
            collection_info['url'] = collection.find_elements_by_tag_name('a')[0].get_attribute('href')
            collection_info['id'] = collection.find_elements_by_class_name('request-title')[0].text.replace(" ", "-").lower()

            print(f"Start get all photo names from {collection_info['name']} collection")
            photos = self.get_photos_of_collection(collection_info['url'])
            collection_info['photos'] = photos
            print(f"{collection_info['name']}: {len(photos)} photos found.")

            collection_infos.append(collection_info)
            with open(os.path.join(conf.scrapped_data_dir, 'collection_info.json'), 'w',
                      encoding='utf-8') as json_file:
                json.dump(collection_infos, json_file)

        return collection_infos

    def get_photos_of_collection(self, url):
        photos = []
        while len(photos) == 0:
            collection_browser = self.init_browser()
            collection_browser.get(url)
            photo_wrappers = self.find_element_by_class_name_until_end(collection_browser, 'photo-wrapper', 5)
            for wrapper in photo_wrappers:
                photo_url = wrapper.find_elements_by_tag_name('a')[0].get_attribute('href')
                if photo_url is not None:
                    photos.append(photo_url.split('/')[-1])
                else:
                    print(f"\n{wrapper.text}\n")
            collection_browser.close()

        return photos

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

        page_content = requests.get(self.page_url, headers = {'User-Agent': UserAgent().opera}).text
        # soup = BeautifulSoup(page_content, "lxml")
        soup = BeautifulSoup(page_content, "html.parser")

        # get Like-count
        like_count_soup = soup.find('h4', {"class": "light like-count"})
        if like_count_soup is not None:
            try:
                self.like_count = int(like_count_soup.text)
            except:
                pass

        # get Photographer info
        photographer_soup = soup.find('div','photographer-info')
        if photographer_soup is not None:
            self.photographer = photographer_soup.find('h4').text

        detail_info_soup = soup.find("section", {"class": "detail-info"})
        if detail_info_soup is not None:
            # get Keywords and caption
            keywork_area_soup = [area for area in detail_info_soup.findAll('div', {"class": "col-md-8"}) if (area.find('h5') is not None and area.find('h5').text.lower().__contains__('keywords'))]
            if len(keywork_area_soup) > 0:
                self.keyworks = [keywork_soup.text.replace(" ", "-").lower() for keywork_soup in keywork_area_soup[0].find_all("a")]

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

