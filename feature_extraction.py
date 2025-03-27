import re
import tldextract
from urllib.parse import urlparse
from collections import Counter
import nltk
from nltk.corpus import words



nltk.download('words')
BRAND_NAMES = set([
    "google", "facebook", "amazon", "apple", "microsoft", "twitter", "instagram",
    "linkedin", "paypal", "dropbox", "alibaba", "ebay", "netflix", "youtube",
    "wordpress", "adobe", "cisco", "oracle", "salesforce", "shopify", "walmart",
    "target", "bestbuy", "costco", "tesla", "uber", "airbnb", "booking", "expedia",
    "spotify", "slack", "zoom", "tiktok", "snapchat", "reddit", "pinterest",
    "whatsapp", "telegram", "discord", "skype", "samsung", "sony", "nike", "adidas",
    "nvidia", "intel", "hp", "dell", "lenovo", "asus", "huawei", "xiaomi", "oneplus",
    "logitech", "seagate", "western digital", "sandisk", "canon", "nikon", "panasonic",
    "philips", "siemens", "bmw", "mercedes", "toyota", "honda", "ford", "volkswagen",
    "chevrolet", "nissan", "hyundai", "kia", "pepsi", "cocacola", "starbucks",
    "mcdonalds", "burgerking", "subway", "dominos", "pizzahut", "kfc", "nintendo",
    "playstation", "xbox", "activision", "ea", "riotgames", "blizzard", "unity",
    "godaddy", "namecheap", "bluehost", "hostgator", "siteground", "cloudflare",
    "akamai", "verisign", "symantec", "mcafee", "avast", "avg", "bitdefender",
    "kaspersky", "norton", "zendesk", "freshdesk", "mailchimp", "sendgrid", "hubspot",
    "hootsuite", "buffer", "semrush", "ahrefs", "moz", "glassdoor", "indeed", "monster",
    "ziprecruiter", "udemy", "coursera", "edx", "khanacademy", "wikipedia", "quora",
    "stackoverflow", "github", "gitlab", "docker", "kubernetes", "jenkins", "ansible",
    "hashicorp", "atlassian"
])

SUSPICIOUS_TLDS = set([
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "online", "site",
    "info", "biz", "cc", "pw", "click", "link", "win", "faith", "party",
    "date", "loan", "stream", "download", "review", "bid", "trade", "accountant",
    "country", "science", "men", "gdn", "racing", "webcam", "video", "chat",
    "mom", "work", "life", "live", "tech", "space", "website", "press", "rest",
    "market", "pub", "social", "rocks", "world", "city", "today", "company",
    "zone", "wiki", "support", "solutions", "email", "network", "center", "pro",
    "store", "shop", "services", "industries", "directory", "foundation",
    "international", "systems", "vision", "community", "cash", "fund", "marketing",
    "media", "money", "name", "report", "school", "eco", "one", "digital", "ltd",
    "group", "institute", "academy", "courses", "events", "agency", "technology",
    "management", "consulting", "properties", "rentals", "vacations", "careers",
    "education", "financial", "healthcare", "house", "land", "mortgage", "energy",
    "engineering", "recipes", "tips", "tools", "training", "university", "villas",
    "airforce", "army", "bargains", "blackfriday", "blue", "build", "builders",
    "cards", "cheap", "christmas", "claims", "cleaning", "clinic", "clothing",
    "coach", "codes", "coffee", "condos", "construction", "contractors", "coupons",
    "credit", "cricket", "dating", "deals", "delivery", "democrat", "dental",
    "discount", "dog", "domains", "equipment", "estate", "exchange", "exposed",
    "fail", "farm", "finance", "fishing", "fit", "flights", "florist", "football",
    "forsale", "furniture", "gallery", "games", "gifts", "glass", "gold", "golf",
    "graphics", "green", "haus", "health", "hockey", "holdings", "holiday",
    "immobilien", "industries", "insure", "kim", "kitchen", "lawyer", "lease",
    "legal", "lgbt", "limited", "limo", "loan", "luxury", "maison", "moda",
    "monster", "nagoya", "ninja", "ong", "organic", "partners", "parts", "photo",
    "photography", "pics", "pictures", "pink", "pizza", "place", "plumbing", "plus"
])

def extract_features(url):
    """
    Extracts 55 features from input data.
    Modify this function based on actual feature extraction logic.
    """

    features = {}
    parsed_url = urlparse(url)
    ext = tldextract.extract(url)
    netloc = parsed_url.netloc

    # f1–2: Full URL length and hostname length
    features['f1'] = len(url)
    features['f2'] = len(netloc)

    # f3: Presence of IP address in hostname
    features['f3'] = 1 if re.match(r"^\d{1,3}(?:\.\d{1,3}){3}$", netloc) else 0

    # f4–20: Count of special characters
    special_chars = ['.', '-', '@', '?', '&', '|', '=', '_', '~', '%', '/', '*', ':', ',', ';', '$', '%20']
    for i, char in enumerate(special_chars, start=4):
        features[f'f{i}'] = url.count(char)

    # f21–24: Count of common terms
    common_terms = ['www', '.com', 'http', '//']
    for i, term in enumerate(common_terms, start=21):
        features[f'f{i}'] = url.count(term)

    # f25: HTTPS token presence
    features['f25'] = 1 if parsed_url.scheme == 'https' else 0

    # f26–27: Ratio of digits in full URL and hostname
    features['f26'] = sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0
    features['f27'] = sum(c.isdigit() for c in netloc) / len(netloc) if len(netloc) > 0 else 0

    # f28: Punycode presence
    features['f28'] = 1 if "xn--" in netloc else 0

    # f29: Port presence
    features['f29'] = 1 if parsed_url.port else 0

    # f30–31: TLD position in path and subdomain
    features['f30'] = 1 if ext.suffix in parsed_url.path else 0
    features['f31'] = 1 if ext.suffix in ext.subdomain else 0

    # f32: Abnormal subdomains
    features['f32'] = 1 if re.search(r"w[w]?[0-9]*", ext.subdomain) else 0

    # f33: Number of subdomains
    features['f33'] = len(ext.subdomain.split('.')) if ext.subdomain else 0

    # f34: Prefix/Suffix presence
    features['f34'] = 1 if '-' in ext.domain else 0

    # f35: Random domains (using English words as a reference)
    english_words = set(words.words())
    features['f35'] = 0 if ext.domain.lower() in english_words else 1

    # f36: Shortening service usage (example check)
    shorteners = {"bit.ly", "goo.gl", "tinyurl.com"}
    features['f36'] = 1 if any(shortener in url for shortener in shorteners) else 0

    # f37: Path extension presence
    malicious_extensions = {'.txt', '.exe', '.js'}
    features['f37'] = 1 if any(url.endswith(extn) for extn in malicious_extensions) else 0

    # f38–39: Placeholder for redirection counts
    features['f38'] = url.count("redirect")
    features['f39'] = url.count("external")

    # f40–47: NLP features based on tokenization of URL
    tokens = re.findall(r'\b\w+\b', url)
    features['f40'] = len(tokens)
    features['f41'] = sum(Counter(token).most_common(1)[0][1] - 1 for token in tokens if len(token) > 1)
    features['f42'] = min([len(token) for token in tokens], default=0)
    features['f43'] = min([len(part) for part in ext.domain.split('-')], default=0)
    features['f44'] = min([len(part) for part in parsed_url.path.split('/') if part], default=0)
    features['f45'] = max([len(token) for token in tokens], default=0)
    features['f46'] = max([len(part) for part in ext.domain.split('-')], default=0)
    features['f47'] = max([len(part) for part in parsed_url.path.split('/') if part], default=0)

    # f48–50: Average word length in tokens, domain parts, and path parts
    features['f48'] = sum(len(token) for token in tokens) / len(tokens) if tokens else 0
    domain_parts = ext.domain.split('-')
    features['f49'] = sum(len(part) for part in domain_parts) / len(domain_parts) if domain_parts else 0
    path_parts = [part for part in parsed_url.path.split('/') if part]
    features['f50'] = sum(len(part) for part in path_parts) / len(path_parts) if path_parts else 0

    # f51: Phish hints count
    phish_hints = {'wp', 'login', 'includes', 'admin', 'content', 'site', 'images', 'js', 'alibaba', 'css', 'myaccount', 'dropbox', 'themes', 'plugins', 'signin', 'view'}
    features['f51'] = sum(url.count(hint) for hint in phish_hints)

    # f52–54: Brand domains presence in domain, subdomain, and path
    features['f52'] = 1 if any(brand in ext.domain for brand in BRAND_NAMES) else 0
    features['f53'] = 1 if any(brand in ext.subdomain for brand in BRAND_NAMES) else 0
    features['f54'] = 1 if any(brand in parsed_url.path for brand in BRAND_NAMES) else 0

    # f55: Suspicious TLD presence
    features['f55'] = 1 if ext.suffix in SUSPICIOUS_TLDS else 0


    return list(features.values())