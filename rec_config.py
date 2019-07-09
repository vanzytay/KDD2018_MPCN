

def get_rec_config(dataset):

    link_ds = ['movie1M','movie100K','imdb', 'movie20M',
    'amazonFood','amazonHome','amazonKindle',
    'amazonAndroid', 'anime',
    'amazonVideo','amazonElectronics','books',
    'netflixPrize','jester','goodbooks','amazonClothes',
    'amazonBeauty','amazonCDS','amazonSports', 'pinterest',
    'lastFM','YahooMusic']

    if(dataset in link_ds):
        data_link = '../deep_cf'
    else:
        data_link = '.'

    if('yelp17' in dataset):
        max_val = 5
        min_val = 1
    elif('gowalla' in dataset):
        max_val = 1
        min_val = 0
    elif('yelp18' in dataset):
        max_val = 5
        min_val = 1
    elif("Amazon" in dataset or 'A2' in dataset):
        max_val = 5
        min_val = 1
    elif(dataset=='imdb'):
        max_val = 10
        min_val = 1
    elif('movie1M' in dataset):
        max_val = 5
        min_val = 0.5
    elif('movie20M' in dataset):
        max_val = 5
        min_val = 0.5
    elif('movie100K' in dataset):
        max_val = 5
        min_val = 1
    elif('pinterest' in dataset):
        max_val = 1
        min_val = 0
    elif('Yahoo' in dataset):
        max_val = 5
        min_val = 1
    elif('amazon' in dataset):
        max_val = 5
        min_val = 0
    elif(dataset=='books'):
        max_val = 10
        min_val = 0
    elif('anime' in dataset):
        max_val = 10
        min_val = 0
    elif('netflixPrize' in dataset):
        max_val = 5
        min_val = 1
    elif('jester' in dataset):
        min_val = -10
        max_val = 10
    elif('goodbooks' in dataset):
        min_val = 1
        max_val = 5
    elif('lastFM' in dataset):
        min_val = 1
        max_val = 10
    return max_val, min_val, data_link
