from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re
import io
import requests
from gtts import gTTS
import nltk
# from googletrans import Translator
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from bs4 import BeautifulSoup
import json

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# Initialize Google Translator
# translator = Translator()

# Initialize IndicBART model for summarization
print("Loading IndicBART model for multilingual summarization...")
try:
    model_name = "ai4bharat/indicbart-ss"
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, use_fast=False, keep_accents=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    use_ai_model = True
    print("IndicBART model loaded successfully")
except Exception as e:
    print(f"Failed to load IndicBART model: {e}")
    print("Falling back to extractive summarization")
    use_ai_model = False



def indicbart_summarize(text, max_length=150, min_length=40):
    """Use IndicBART for Tamil text summarization"""
    try:
        # Tokenize input text
        inputs = tokenizer(text.strip(), return_tensors="pt", max_length=512, truncation=True, padding=True)
        
        # Generate summary with better parameters
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                min_length=min_length,
                length_penalty=1.5,
                num_beams=6,
                early_stopping=True,
                no_repeat_ngram_size=3,
                do_sample=False,
                temperature=1.0
            )
        
        # Decode summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Post-process to ensure proper sentence structure
        summary = post_process_summary(summary, text.strip())
        return summary
        
    except Exception as e:
        print(f"IndicBART summarization failed: {e}")
        # Fallback to extractive
        return extractive_summary(text)

def post_process_summary(summary, original_text):
    """Post-process summary to ensure proper context and sentence boundaries"""
    try:
        # Remove incomplete sentences at the beginning
        sentences = re.split(r'[.!?।]+', summary.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if not sentences:
            return extractive_summary(original_text)
        
        # Ensure first sentence starts properly (not mid-sentence)
        first_sentence = sentences[0]
        
        # Check if first sentence seems incomplete (starts with lowercase or common conjunctions)
        tamil_conjunctions = ['மற்றும்', 'அல்லது', 'ஆனால்', 'எனவே', 'இருப்பினும்']
        english_conjunctions = ['and', 'but', 'or', 'so', 'however', 'therefore']
        
        words = first_sentence.split()
        if words and (words[0].lower() in tamil_conjunctions + english_conjunctions or 
                     (words[0][0].islower() and words[0] not in ['இன்று', 'நேற்று', 'நாளை'])):
            # Remove incomplete first sentence
            sentences = sentences[1:]
        
        if not sentences:
            return extractive_summary(original_text)
        
        # Reconstruct summary with proper punctuation
        processed_summary = '. '.join(sentences)
        if not processed_summary.endswith(('.', '!', '?', '।')):
            processed_summary += '.'
            
        return processed_summary
        
    except Exception as e:
        print(f"Post-processing failed: {e}")
        return summary

def extractive_summary(text, target_compression=0.5):
    """Fallback extractive summarization"""
    sentences = re.split(r'[.!?।]+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
    
    if len(sentences) <= 2:
        return text
    
    # Simple frequency-based scoring
    all_words = []
    for sentence in sentences:
        words = re.findall(r'\b\w+\b', sentence.lower())
        all_words.extend(words)
    
    word_freq = {}
    for word in all_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    sentence_scores = []
    for i, sentence in enumerate(sentences):
        words = re.findall(r'\b\w+\b', sentence.lower())
        score = sum(word_freq[word] for word in words)
        sentence_scores.append((score, i, sentence))
    
    # Sort by score and select top sentences
    sentence_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Target length
    target_length = int(len(text) * target_compression)
    selected = []
    current_length = 0
    
    for score, idx, sentence in sentence_scores:
        if current_length + len(sentence) <= target_length:
            selected.append((idx, sentence))
            current_length += len(sentence)
    
    # Sort by original order
    selected.sort(key=lambda x: x[0])
    return '. '.join([sentence for _, sentence in selected]) + '.'

def is_tamil_text(text):
    """Check if text contains Tamil content (very permissive for mixed content)"""
    if not text or len(text.strip()) < 10:
        return False
    
    # Tamil Unicode range: 0x0B80-0x0BFF
    tamil_chars = sum(1 for char in text if '\u0B80' <= char <= '\u0BFF')
    
    # Very permissive - allow mixed content with minimal Tamil presence
    return tamil_chars > 1  # Just 2+ Tamil characters needed

def filter_tamil_content(text):
    """Filter and clean Tamil content, removing English and metadata"""
    # Initial cleaning - remove obvious metadata blocks
    text = re.sub(r'\b(advertisement|sponsored content|promoted post)\b.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(tags?|categories?)\s*:.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(share|follow|subscribe)\s*:.*?\n', '', text, flags=re.IGNORECASE)
    
    # Split into sentences
    sentences = re.split(r'[.!?।\n]+', text)
    tamil_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        
        # Skip very short sentences
        if len(sentence) < 20:
            continue
            
        # Enhanced metadata patterns
        metadata_patterns = [
            # Copyright and legal
            r'\b(copyright|©|all rights reserved|terms of use|privacy policy)\b',
            # Social media and engagement
            r'\b(subscribe|follow|share|like|comment|join|register|login|sign up)\b',
            r'\b(facebook|twitter|instagram|youtube|whatsapp|telegram|linkedin)\b',
            # Advertisements and promotions
            r'\b(advertisement|sponsored|promoted|ad\s|banner|popup)\b',
            # Navigation and UI elements
            r'\b(read more|click here|view all|see more|load more|show more)\b',
            r'\b(home|menu|search|contact|about|help|support)\b',
            r'\b(next|previous|back|forward|up|down|left|right)\b',
            # Publishing metadata
            r'\b(updated|published|posted|edited|modified|created)\s*(on|at|by)',
            r'\b(author|reporter|correspondent|editor|writer|journalist)\s*:',
            r'\b(source|credit|photo|image|video)\s*:',
            # Technical elements
            r'\b(email|website|www\.|http|https|ftp|url)\b',
            r'\b(download|upload|install|update|version)\b',
            # News metadata
            r'^(breaking|live|update|alert)\s*:',
            r'^(tags?|categories?|related|similar)\s*:',
            r'\b(breaking news|live updates|news alert)\b',
            # Date/time patterns
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{1,2}:\d{2}\s*(am|pm|AM|PM)\b',
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            # Common website elements
            r'\b(cookie|session|cache|browser|device)\b',
            r'\b(notification|alert|popup|modal|dialog)\b',
            # Rating and feedback
            r'\b(rating|review|feedback|comment|reply)\b',
            # Location/contact info
            r'\b(address|phone|mobile|email|contact)\s*:',
            # Subscription/membership
            r'\b(premium|subscription|membership|account)\b'
        ]
        
        # Skip if contains metadata patterns
        if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in metadata_patterns):
            continue
            
        # Allow mixed content - only skip if no Tamil characters at all
        tamil_chars = sum(1 for char in sentence if '\u0B80' <= char <= '\u0BFF')
        if tamil_chars == 0:
            continue
            
        # Skip navigation/menu items
        if ':' in sentence and len(sentence) < 60:
            continue
            
        # Skip if contains too many numbers, symbols, or special characters
        non_alpha = sum(1 for char in sentence if not char.isalpha() and not char.isspace())
        if non_alpha > len(sentence) * 0.25:
            continue
            
        # Skip sentences with excessive punctuation
        punct_count = sum(1 for char in sentence if char in '.,;:!?()[]{}"\'')
        if punct_count > len(sentence) * 0.15:
            continue
            
        # Skip if sentence is mostly uppercase (likely headers/titles)
        if sum(1 for char in sentence if char.isupper()) > len(sentence) * 0.3:
            continue
            
        # Skip common website boilerplate text
        boilerplate_keywords = [
            'வெബ்சைட்', 'இணையதளம்', 'பதிவு', 'உள்நுழை', 'பதிவிறக்க',
            'பகிர்', 'சந்தா', 'விளம்பர', 'குக்கீ', 'அறிவிப்பு'
        ]
        if any(keyword in sentence for keyword in boilerplate_keywords):
            continue
            
        tamil_sentences.append(sentence)
    
    # Join filtered sentences
    filtered_text = '. '.join(tamil_sentences)
    
    # Final cleaning
    filtered_text = re.sub(r'\s+', ' ', filtered_text).strip()
    filtered_text = re.sub(r'\s*\.\s*\.+', '.', filtered_text)  # Remove multiple dots
    filtered_text = re.sub(r'^[.\s]+|[.\s]+$', '', filtered_text)  # Trim dots and spaces
    
    return filtered_text

def scrape_article_text(url):
    """Extract Tamil article text from URL with enhanced robustness"""
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'ta,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=20, verify=False, allow_redirects=True)
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.reason}")
    except requests.exceptions.Timeout:
        raise Exception("Request timeout - website took too long to respond")
    except requests.exceptions.ConnectionError:
        raise Exception("Connection failed - unable to reach the website")
    except Exception as e:
        raise Exception(f"Network error: {str(e)}")
    
    try:
        # Better encoding detection
        if response.encoding is None or response.encoding.lower() in ['iso-8859-1', 'ascii']:
            response.encoding = 'utf-8'
        
        soup = BeautifulSoup(response.content, 'html.parser', from_encoding=response.encoding)
        
        # Remove unwanted elements more aggressively
        unwanted_tags = ['script', 'style', 'noscript', 'iframe', 'embed', 'object', 'form', 'input', 'button', 'select', 'textarea']
        for tag in unwanted_tags:
            for element in soup.find_all(tag):
                element.decompose()
        
        # Remove by class/id patterns
        unwanted_patterns = [
            'ad', 'ads', 'advertisement', 'sponsored', 'promo', 'banner',
            'social', 'share', 'sharing', 'follow', 'subscribe', 'newsletter',
            'comment', 'comments', 'feedback', 'rating', 'review',
            'menu', 'navigation', 'nav', 'navbar', 'sidebar', 'header', 'footer',
            'tag', 'category', 'related', 'similar', 'widget', 'plugin'
        ]
        
        for pattern in unwanted_patterns:
            for element in soup.find_all(attrs={'class': lambda x: x and any(pattern in str(c).lower() for c in x)}):
                element.decompose()
            for element in soup.find_all(attrs={'id': lambda x: x and pattern in str(x).lower()}):
                element.decompose()
        
        # Enhanced article content extraction
        content_candidates = []
        
        # Method 1: Semantic HTML5 article tags
        articles = soup.find_all('article')
        for article in articles:
            text = article.get_text(separator=' ', strip=True)
            if len(text) > 100:
                content_candidates.append(('article', text, len(text)))
        
        # Method 2: Common content selectors
        content_selectors = [
            '.article-content', '.article-body', '.post-content', '.post-body',
            '.entry-content', '.story-content', '.news-content', '.main-content',
            '.content', '#content', '#main', '#article', 'main'
        ]
        
        for selector in content_selectors:
            elements = soup.select(selector)
            for elem in elements:
                text = elem.get_text(separator=' ', strip=True)
                if len(text) > 100:
                    content_candidates.append((selector, text, len(text)))
        
        # Method 3: Paragraph aggregation
        paragraphs = soup.find_all('p')
        if paragraphs:
            para_text = ' '.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20])
            if len(para_text) > 100:
                content_candidates.append(('paragraphs', para_text, len(para_text)))
        
        # Method 4: Div content extraction
        divs = soup.find_all('div')
        for div in divs:
            # Skip divs that are likely containers
            if div.find_all(['div', 'article', 'section']):
                continue
            text = div.get_text(separator=' ', strip=True)
            if len(text) > 200:
                content_candidates.append(('div', text, len(text)))
        
        # Evaluate candidates for Tamil content
        best_content = ""
        best_score = 0
        
        for source, raw_text, length in content_candidates:
            # Filter and clean the content
            filtered_text = filter_tamil_content(raw_text)
            
            if len(filtered_text) < 50:
                continue
            
            # Calculate quality score (more permissive)
            tamil_chars = sum(1 for char in filtered_text if '\u0B80' <= char <= '\u0BFF')
            total_chars = len([char for char in filtered_text if char.isalpha()])
            tamil_ratio = tamil_chars / total_chars if total_chars > 0 else 0
            
            # Score based on Tamil presence and content length (reduced threshold)
            score = (tamil_ratio * 0.7 + 0.3) * min(len(filtered_text) / 500, 1.0)
            
            if score > best_score and tamil_ratio > 0.1:  # Much lower threshold
                best_score = score
                best_content = filtered_text
        
        # Final fallback: extract all text and filter
        if not best_content or len(best_content) < 50:
            all_text = soup.get_text(separator=' ', strip=True)
            if all_text:
                filtered_all = filter_tamil_content(all_text)
                if len(filtered_all) > 50:
                    best_content = filtered_all
        
        # Validate final content
        if not best_content or len(best_content.strip()) < 50:
            raise Exception("Could not extract sufficient Tamil article content from the webpage")
        
        # Final cleaning
        best_content = re.sub(r'\s+', ' ', best_content).strip()
        best_content = re.sub(r'([.!?])\s*([.!?])+', r'\1', best_content)
        
        # Ensure minimum content quality
        if not is_tamil_text(best_content):
            raise Exception("Extracted content does not contain sufficient Tamil text")
        
        return best_content
        
    except Exception as e:
        if "Could not extract sufficient Tamil" in str(e) or "does not contain sufficient Tamil" in str(e):
            raise Exception(str(e))
        else:
            raise Exception(f"Failed to scrape article: {str(e)}")



def fallback_translate(text):
    """Simple fallback translation using word mappings"""
    tamil_to_english = {
        'செய்தி': 'news', 'அரசு': 'government', 'மக்கள்': 'people',
        'நாடு': 'country', 'உலகம்': 'world', 'பணம்': 'money',
        'வேலை': 'work', 'கல்வி': 'education', 'மருத்துவம்': 'medicine',
        'விளையாட்டு': 'sports', 'திரைப்படம்': 'movie', 'இன்று': 'today',
        'நேற்று': 'yesterday', 'நாளை': 'tomorrow', 'முக்கியம்': 'important',
        'புதிய': 'new', 'பழைய': 'old', 'பெரிய': 'big', 'சிறிய': 'small'
    }
    
    words = text.split()
    translated_words = []
    
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word)
        if clean_word in tamil_to_english:
            translated_words.append(tamil_to_english[clean_word])
        else:
            translated_words.append(word)
    
    return ' '.join(translated_words)

def google_translate_api(text, source_lang='ta', target_lang='en'):
    """Translate using Google Translate API via HTTP request"""
    if not text or not text.strip():
        return ""
        
    # Limit text length
    if len(text) > 5000:
        text = text[:5000]
    
    try:
        import urllib.parse
        encoded_text = urllib.parse.quote(text)
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={source_lang}&tl={target_lang}&dt=t&q={encoded_text}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result and result[0] and result[0][0] and result[0][0][0]:
                translated = ''.join([item[0] for item in result[0] if item[0]])
                print(f"Google Translate successful: {translated[:50]}...")
                return translated
        
        raise Exception("Google Translate API failed")
        
    except Exception as e:
        print(f"Google Translate API error: {e}")
        raise e

def bing_translate_api(text, source_lang='ta', target_lang='en'):
    """Translate using Bing Translator API"""
    if not text or not text.strip():
        return ""
        
    try:
        import urllib.parse
        encoded_text = urllib.parse.quote(text)
        url = f"https://www.bing.com/ttranslatev3?isVertical=1&&IG=1234567890AB&IID=translator.5028.1"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'fromLang': source_lang,
            'toLang': target_lang,
            'text': text
        }
        
        response = requests.post(url, headers=headers, data=data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result and result[0] and 'translations' in result[0] and result[0]['translations']:
                translated = result[0]['translations'][0]['text']
                print(f"Bing Translate successful: {translated[:50]}...")
                return translated
        
        raise Exception("Bing Translate API failed")
        
    except Exception as e:
        print(f"Bing Translate API error: {e}")
        raise e

def mymemory_translate_api(text, source_lang='ta', target_lang='en'):
    """Translate using MyMemory API"""
    if not text or not text.strip():
        return ""
        
    try:
        url = 'https://api.mymemory.translated.net/get'
        params = {
            'q': text,
            'langpair': f'{source_lang}|{target_lang}'
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result and 'responseData' in result and 'translatedText' in result['responseData']:
                translated = result['responseData']['translatedText']
                print(f"MyMemory Translate successful: {translated[:50]}...")
                return translated
        
        raise Exception("MyMemory API failed")
        
    except Exception as e:
        print(f"MyMemory API error: {e}")
        raise e

def translate_text_with_service(text, source_lang='ta', target_lang='en'):
    """Translation priority: MyMemory -> Bing (long text) -> Google -> Fallback"""
    if not text or not text.strip():
        return "", "None"
    
    # Try MyMemory first (best for short text)
    if len(text) <= 500:
        try:
            result = mymemory_translate_api(text, source_lang, target_lang)
            return result, "MyMemory API"
        except Exception as e:
            print(f"MyMemory failed: {e}")
    
    # Try Bing for longer text or if MyMemory failed
    try:
        result = bing_translate_api(text, source_lang, target_lang)
        return result, "Bing Translator"
    except Exception as e:
        print(f"Bing Translate failed: {e}")
    
    # Try Google Translate as fallback
    try:
        result = google_translate_api(text, source_lang, target_lang)
        return result, "Google Translate"
    except Exception as e:
        print(f"Google Translate failed: {e}")
    
    # Final fallback to word mapping
    print("All translation services failed, using word mapping fallback")
    return fallback_translate(text), "Fallback Dictionary"

def translate_text(text, source_lang='ta', target_lang='en'):
    """Simple wrapper for backward compatibility"""
    result, _ = translate_text_with_service(text, source_lang, target_lang)
    return result

app = FastAPI(title='Tamil News Summarizer API - IndicBART', version='2.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NewsInput(BaseModel):
    text: str
    max_length: int = 150
    min_length: int = 40

class TranslateInput(BaseModel):
    original_text: str
    summary_text: str

class TranslateSummaryInput(BaseModel):
    summary_text: str

class TTSInput(BaseModel):
    text: str
    language: str = 'ta'

class ChatInput(BaseModel):
    article_text: str
    question: str
    language: str = 'ta'
    transliterate: bool = False



@app.get("/")
def root():
    return {
        "message": "Tamil News Summarizer API with IndicBART",
        "model": "IndicBART (ai4bharat/indicbart-ss)" if use_ai_model else "Extractive Fallback",
        "features": ["Summarization", "Translation (Google Translate)", "Text-to-Speech", "Q&A Chatbot", "Transliteration"],
        "status": "ready"
    }

@app.post('/summarize')
def summarize(news: NewsInput):
    try:
        if not news.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Use IndicBART if available, otherwise extractive
        if use_ai_model:
            summary = indicbart_summarize(news.text, news.max_length, news.min_length)
            mode = "IndicBART"
        else:
            summary = extractive_summary(news.text, target_compression=0.45)
            mode = "Extractive"
        
        # Auto-translate using multiple services
        original_english = None
        summary_english = None
        translation_service = None
        
        try:
            print(f"Attempting to translate original text: {news.text[:100]}...")
            original_english, service1 = translate_text_with_service(news.text)
            print(f"Original translation successful: {original_english[:100] if original_english else 'None'}...")
        except Exception as e:
            print(f"Original text translation failed: {e}")
            
        try:
            print(f"Attempting to translate summary: {summary[:100]}...")
            summary_english, service2 = translate_text_with_service(summary)
            translation_service = service2
            print(f"Summary translation successful: {summary_english[:100] if summary_english else 'None'}...")
        except Exception as e:
            print(f"Summary translation failed: {e}")
        
        response = {
            'original': news.text,
            'summary': summary,
            'stats': {
                'original_length': len(news.text),
                'summary_length': len(summary),
                'compression_ratio': round((1 - len(summary) / len(news.text)) * 100, 1)
            },
            'mode': mode
        }
        
        # Add translations if successful
        if original_english and summary_english:
            response['original_english'] = original_english
            response['summary_english'] = summary_english
            response['translation_service'] = translation_service
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")



@app.post('/translate')
def translate(translate_input: TranslateInput):
    try:
        if not translate_input.original_text.strip() or not translate_input.summary_text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        original_english, service1 = translate_text_with_service(translate_input.original_text)
        summary_english, service2 = translate_text_with_service(translate_input.summary_text)
        
        return {
            'original_tamil': translate_input.original_text,
            'summary_tamil': translate_input.summary_text,
            'original_english': original_english,
            'summary_english': summary_english,
            'translation_method': service2,
            'status': 'success'
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.post('/translate-summary')
def translate_summary(translate_input: TranslateSummaryInput):
    try:
        if not translate_input.summary_text.strip():
            raise HTTPException(status_code=400, detail="Summary text cannot be empty")
        
        summary_english, service = translate_text_with_service(translate_input.summary_text)
        
        return {
            'summary_tamil': translate_input.summary_text,
            'summary_english': summary_english,
            'translation_method': service,
            'status': 'success'
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.post('/tts')
def text_to_speech(tts_input: TTSInput):
    try:
        if not tts_input.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        tts = gTTS(text=tts_input.text, lang=tts_input.language, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(audio_buffer.read()),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")

def extract_relevant_info(article_text, question):
    """Extract relevant information from article based on question"""
    sentences = re.split(r'[.!?।]+', article_text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    question_words = set(re.findall(r'\b\w+\b', question.lower()))
    
    relevant_sentences = []
    for sentence in sentences:
        sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
        overlap = len(question_words.intersection(sentence_words))
        if overlap > 0:
            relevant_sentences.append((overlap, sentence))
    
    relevant_sentences.sort(key=lambda x: x[0], reverse=True)
    return [sentence for _, sentence in relevant_sentences[:3]]

def transliterate_to_tamil(english_text):
    """Fallback transliteration using word mappings"""
    fallback_mappings = {
        'yaar': 'யார்', 'enna': 'என்ன', 'enga': 'எங்க', 'eppo': 'எப்போ',
        'ethu': 'எது', 'pesinar': 'பேசினார்', 'sonnaar': 'சொன்னார்',
        'nadanthathu': 'நடந்தது', 'mukkiyam': 'முக்கியம்', 'seythi': 'செய்தி'
    }
    
    words = english_text.lower().split()
    result = []
    
    for word in words:
        if word in fallback_mappings:
            result.append(fallback_mappings[word])
        else:
            result.append(word)
    
    return ' '.join(result)



def generate_answer(article_text, question, language='ta'):
    """Generate answer based on article content"""
    try:
        relevant_info = extract_relevant_info(article_text, question)
        
        if not relevant_info:
            if language == 'ta':
                return "மன்னிக்கவும், இந்த கேள்விக்கான பதில் கட்டுரையில் கிடைக்கவில்லை."
            else:
                return "Sorry, I couldn't find relevant information in the article to answer this question."
        
        answer = '. '.join(relevant_info[:2])
        
        if language == 'ta':
            prefix = "கட்டுரையின் அடிப்படையில்: "
        else:
            prefix = "Based on the article: "
            
        return prefix + answer
        
    except Exception as e:
        if language == 'ta':
            return "மன்னிக்கவும், பதில் உருவாக்குவதில் பிழை ஏற்பட்டது."
        else:
            return "Sorry, there was an error generating the answer."

@app.post('/chat')
def chat_with_article(chat_input: ChatInput):
    try:
        if not chat_input.article_text.strip() or not chat_input.question.strip():
            raise HTTPException(status_code=400, detail="Article text and question cannot be empty")
        
        processed_question = chat_input.question
        if chat_input.transliterate:
            processed_question = transliterate_to_tamil(chat_input.question)
        
        answer = generate_answer(chat_input.article_text, processed_question, chat_input.language)
        
        return {
            'original_question': chat_input.question,
            'processed_question': processed_question,
            'answer': answer,
            'language': chat_input.language,
            'transliterated': chat_input.transliterate,
            'status': 'success'
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")



@app.post('/transliterate')
def transliterate_text(text_input: dict):
    try:
        english_text = text_input.get('text', '')
        if not english_text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        tamil_text = transliterate_to_tamil(english_text)
        
        return {
            'english': english_text,
            'tamil': tamil_text,
            'status': 'success'
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transliteration failed: {str(e)}")