import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
import time


class CollectionScraper:
    def __init__(self):
        self.collection_url = 'https://community.snapwire.co/collections'
        self.collections = []
        self.browser_path = './chromedriver.exe'

    def get_collection(self):
        browser = webdriver.Chrome(executable_path=self.browser_path)
        browser.get(self.collection_url)
        self.collections = browser.find_elements_by_class_name('request-title')

        while (True):
            browser.execute_script("window.scrollTo(0, 5000);")
            time.sleep(5)
            new_collections = browser.find_elements_by_class_name('request-title')
            if len(new_collections) != len(self.collections):
                self.collections = new_collections
            else:
                break

        self.collections = [collection_name.text for collection_name in self.collections]
        print(self.collections)


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

    def get_detail(self):

        page_content = requests.get(self.page_url).text

        soup = BeautifulSoup(page_content, "lxml")

        # get Like-count
        like_count_soup = soup.find('h4', {"class": "light like-count"})
        self.like_count = int(like_count_soup.text)

        # get Photographer info
        photographer_soup = soup.find('div','photographer-info').find('h4')
        self.photographer = photographer_soup.text

        detail_info_soup = soup.find("section", {"class": "detail-info"})
        # get Keywords and caption
        keywork_area_soup = [area for area in detail_info_soup.findAll('div', {"class": "col-md-8"}) if (area.find('h5') is not None and area.find('h5').text.lower().__contains__('keywords'))]
        if len(keywork_area_soup) > 0:
            self.keyworks = [keywork_soup.text for keywork_soup in keywork_area_soup[0].find_all("a")]

        caption = detail_info_soup.find('p', {'id':'caption-text'}).text
        caption = caption.encode('ascii', 'ignore').decode('utf-8')
        self.caption = caption.strip()

        # get Photo URL:
        photo_url_soup = soup.find('img', {"class": "margin-bottom"})
        self.photo_url = photo_url_soup.attrs['src']
        # get Photo detail

