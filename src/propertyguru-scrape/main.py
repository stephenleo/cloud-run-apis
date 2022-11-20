import os
from datetime import datetime
from locale import LC_NUMERIC, atof, setlocale
from time import sleep

setlocale(LC_NUMERIC, '')

import pandas as pd
import uvicorn
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.logger import logger
from seleniumbase import Driver
from tqdm import tqdm

base_url = "https://www.propertyguru.com.sg"

if os.name == 'posix':  #Check if this is linux
    from pyvirtualdisplay import Display
    display = Display()
    display.start()

# Instantiate the app
app = FastAPI()

@app.get("/")
async def life_check():
    return {"status": "ok"}

# Scrape Endpoint
@app.post("/scrape")
async def scrape_url(url: str, dest_table: str) -> dict:
    driver = Driver(headless=True, uc=True)
    data = {
        "id": [], "title": [], "href": [], "hero_img": [], 
        "address": [], "price": [], "floor_area_sqft": [], "psf": []
    }

    # Pagination
    def generator(url):
        while url:
            driver.get(f"{base_url}{url}")

            html = driver.page_source
            html_soup = BeautifulSoup(html, 'html.parser')

            listings = html_soup.find_all("div", class_="listing-card")

            for listing in listings:
                nav_link = listing.find_all("a", class_="nav-link")
                data["id"].append(int(nav_link[-1].get("data-listing-id", "")))
                data["title"].append(nav_link[-3].text)
                data["href"].append(nav_link[-3]["href"])
                data["hero_img"].append(nav_link[-2].img["content"] if nav_link[-2].img else "")

                data["address"].append(listing.find_all("span", itemprop="streetAddress")[0].text)
                data["price"].append(atof(listing.find_all("span", class_="price")[0].text.replace(",", "")))
                data["floor_area_sqft"].append(atof(listing.find_all("li", class_="listing-floorarea")[0].text.split()[0]))
                data["psf"].append(atof(listing.find_all("li", class_="listing-floorarea")[1].text.split(u"\xa0")[1].replace(",", "")))

            next_button = html_soup.find_all("li", class_="pagination-next")[0].a
            url = next_button["href"] if next_button else ""
            sleep(10)

            yield

    for _ in tqdm(generator(url)): pass

    # Convert to Dataframe
    df = pd.DataFrame(data)
    df["date"] = datetime.now()

    logger.info(df.shape)

    # Save to BQ
    df.to_gbq(
        destination_table = f"leo-gcp-sanbox.hdb_prices.{dest_table}",
        if_exists = "append",
        progress_bar = True
    )

    driver.quit()

    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    