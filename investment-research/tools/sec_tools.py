from sec_edgar_downloader import Downloader

def download_10k(ticker):

    dl = Downloader("sec_filings")

    dl.get("10-K", ticker, limit=1)

    return "10-K downloaded"