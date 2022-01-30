# import bs4
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup

test_url = "https://www.bbc.co.uk/news/uk-england-london-55730459"
# 'https://www.dailymail.co.uk/news/article-9038157/Coronavirus-Europe-UK-tourists-banned-EU-Brexit-pandemic-travel-rules.html'
# 'https://www.foxnews.com/politics/hunter-bidens-china-business-deals-leading-up-to-2018-probe-detailed-in-senate-report'
# 'https://www.bbc.co.uk/news/uk-55086621'
# 'https://www.thesun.co.uk/news/brexit/13435228/britain-eu-restart-brexit-talks-sunday-deadline-boris/'
# "https://www.foxnews.com/entertainment/obama-rips-fox-news-viewers-you-are-living-on-a-different-planet"


def scrape(my_url):
    # Open connection and grab page
    u_client = uReq(my_url)
    page_html = u_client.read()
    u_client.close()
    # print()
    # print("Website :", my_url[12:18])
    # print()
    page_soup = soup(page_html, "html.parser")

    # Fetch headline and body
    body_text = ""
    if my_url[12:15] == 'bbc':
        h_containers = page_soup.findAll("h1", {"id": "main-heading"})
        b_containers = page_soup.findAll("div", {"data-component": "text-block"})
        # b_containers.extend(page_soup.findAll("div", {"a" : "css-yidnqd-InlineLink"}))  #if including hyperlinks
    elif my_url[12:15] == 'fox':
        h_containers = page_soup.findAll("h1", {"class": "headline"})
        b_containers = page_soup.findAll("p")  # , {"class": "speakable"})
    elif my_url[12:15] == 'dai':
        h_containers = page_soup.findAll("h2")  # , {"class": "headline"})
        b_containers = page_soup.findAll("p", {"class": "mol-para-with-font"})
    elif my_url[12:18] == 'thesun':
        h_containers = page_soup.findAll("h1", {"class": "article__headline"})
        b_containers = page_soup.findAll("p")  # , {"class": "speakable"})
    # not tested
    elif my_url[12:15] == 'edi':
        h_containers = page_soup.findAll("h1", {"class": "pg-headline"})
        b_containers = page_soup.findAll("div", {"class": "zn-body__paragraph"})
    elif my_url[12:18] == 'thegua':
        h_containers = page_soup.findAll("h1")
        b_containers = page_soup.findAll("p")
    elif my_url[12:15] == 'sta':
        h_containers = page_soup.findAll("h1")
        b_containers = page_soup.findAll("p")
    else:
        print("website cannot be scraped")
        return

    # format data
    headline_text = h_containers[0].text
    # print("Headline { " + headline_text + " } ")
    # print()
    # print("Article { ")
    for t in b_containers:
        # print(t.text)
        body_text += t.text + " "
    # print(" } ")
    # print()
    return headline_text, body_text

# headline_text, body_text = scrape(test_url)

# # uncomplete generic scraping function
# def general_scrape(my_url):
#     # Open connection and grab page
#     u_client = uReq(my_url)
#     page_html = u_client.read()
#     u_client.close()
#     print()
#     print("Website :", my_url[12:18])
#     print()
#     page_soup = soup(page_html, "html.parser")
#     # Fetch headline and body
#     body_text = ""
#     h_containers = page_soup.findAll("h1", {"id": "main-heading"})  # h1 or h2
#     b_containers = page_soup.findAll("div", {"data-component": "text-block"})  # div or p
#     # format data
#     headline_text = h_containers[0].text
#     print("Headline { " + headline_text + " } ")
#     print()
#     print("Article { ")
#     for t in b_containers:
#         print(t.text)
#         body_text += t.text + " "
#     print(" } ")
#     print()
#     return headline_text, body_text


# other scraping methods
# page_soup.h1
# page_soup.body.div
# containers = page_soup.findAll("div", {"data-entityid" : "container-top-stories#2"})
# container[0].h3.text
# headlineContainers = page_soup.findAll("h1")
