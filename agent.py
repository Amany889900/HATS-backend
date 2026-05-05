# -*- coding: utf-8 -*-
"""
Misinformation Detection Agent - FastAPI Service
"""
 
import os
import json
import re
import requests
import tldextract
from urllib.parse import urlparse
from datetime import datetime, timedelta
from typing import List, Dict, Optional
 
from groq import Groq
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
 
# ---------------------------------------------------------------------------
# Configuration  (override via environment variables in production)
# ---------------------------------------------------------------------------
GROQ_API_KEY   = os.getenv("GROQ_API_KEY",   "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")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "ee285149e0f2ce50f74d1dae254489bbce93b141")
SELECTED_MODEL = os.getenv("GROQ_MODEL",     "openai/gpt-oss-120b")
 
 
# ===========================================================================
# Source Credibility Scorer
# ===========================================================================
class SourceCredibilityScorer:
    """Assess the credibility of web sources based on domain authority,
    source type, and content characteristics."""
 
    HIGH_TRUST_DOMAINS = {
        '.gov', '.gov.uk', '.gov.au', '.gov.ca', '.gov.in',
        '.edu', '.ac.uk', '.edu.au', '.edu.ca',
        '.int', 'who.int', 'un.org', 'worldbank.org', 'imf.org',
        'reuters.com', 'apnews.com', 'bbc.com', 'bbc.co.uk', 'npr.org',
        'wsj.com', 'nytimes.com', 'washingtonpost.com', 'economist.com',
        'ft.com', 'bloomberg.com', 'cnn.com', 'nbcnews.com', 'abcnews.go.com',
        'cbsnews.com', 'usatoday.com', 'latimes.com', 'chicagotribune.com',
        'nature.com', 'science.org', 'sciencedirect.com', 'springer.com',
        'plos.org', 'biorxiv.org', 'medrxiv.org', 'arxiv.org',
        'pubmed.ncbi.nlm.nih.gov', 'ncbi.nlm.nih.gov', 'scholar.google.com',
        'jstor.org', 'ieee.org', 'acm.org',
        'cdc.gov', 'nih.gov', 'mayoclinic.org', 'clevelandclinic.org',
        'hopkinsmedicine.org', 'health.harvard.edu', 'medscape.com',
        'webmd.com', 'healthline.com', 'medicalnewstoday.com',
        'snopes.com', 'factcheck.org', 'politifact.com', 'leadstories.com',
        'afp.com', 'reuters.com/fact-check',
    }
 
    MEDIUM_TRUST_DOMAINS = {
        'usnews.com', 'newsweek.com', 'time.com', 'forbes.com', 'businessinsider.com',
        'vox.com', 'theatlantic.com', 'newyorker.com', 'harpers.org',
        'theguardian.com', 'independent.co.uk', 'telegraph.co.uk',
        'smh.com.au', 'theage.com.au', 'theglobeandmail.com',
        'wired.com', 'techcrunch.com', 'theverge.com', 'arstechnica.com',
        'engadget.com', 'cnet.com', 'zdnet.com', 'gizmodo.com',
        'wikipedia.org', 'britannica.com', 'thoughtco.com',
    }
 
    LOW_TRUST_DOMAINS = {
        'facebook.com', 'twitter.com', 'x.com', 'instagram.com', 'tiktok.com',
        'linkedin.com', 'reddit.com', 'pinterest.com', 'snapchat.com',
        'medium.com', 'wordpress.com', 'blogger.com', 'tumblr.com',
        'wix.com', 'weebly.com', 'squarespace.com', 'substack.com',
        'buzzfeed.com', 'distractify.com', 'upworthy.com', 'viralnova.com',
        'shared.com', 'thoughtcatalog.com',
        'naturalnews.com', 'infowars.com', 'breitbart.com', 'dailymail.co.uk',
        'dailystar.co.uk', 'thesun.co.uk', 'nypost.com',
        'zerohedge.com', 'rt.com', 'sputniknews.com',
    }
 
    @classmethod
    def extract_domain(cls, url: str) -> str:
        try:
            extracted = tldextract.extract(url)
            return f"{extracted.domain}.{extracted.suffix}".lower()
        except Exception:
            parsed = urlparse(url)
            return parsed.netloc.lower().replace('www.', '')
 
    @classmethod
    def assess_credibility(cls, url: str, snippet: str = "", title: str = "") -> Dict:
        domain = cls.extract_domain(url)
        credibility_score = 0.50
        source_type = "unknown"
        reasoning: List[str] = []
        domain_checked = False
 
        for trusted in cls.HIGH_TRUST_DOMAINS:
            if trusted in domain or domain.endswith(trusted):
                credibility_score = 0.85
                source_type = "highly_trusted"
                reasoning.append(f"Domain {domain} is in high-trust list")
                domain_checked = True
                break
 
        if not domain_checked:
            for trusted in cls.MEDIUM_TRUST_DOMAINS:
                if trusted in domain or domain.endswith(trusted):
                    credibility_score = 0.65
                    source_type = "moderately_trusted"
                    reasoning.append(f"Domain {domain} is in medium-trust list")
                    domain_checked = True
                    break
 
        if not domain_checked:
            for low in cls.LOW_TRUST_DOMAINS:
                if low in domain or domain.endswith(low):
                    credibility_score = 0.30
                    source_type = "low_trust"
                    reasoning.append(f"Domain {domain} is in low-trust list")
                    domain_checked = True
                    break
 
        if not domain_checked:
            tld = domain.split('.')[-1] if '.' in domain else ''
            if tld == 'gov':
                credibility_score, source_type = 0.95, "government"
                reasoning.append("Government domain (.gov)")
            elif tld == 'edu':
                credibility_score, source_type = 0.90, "educational"
                reasoning.append("Educational domain (.edu)")
            elif tld == 'org' and 'who' in domain:
                credibility_score, source_type = 0.90, "international_org"
                reasoning.append("International organization")
            elif tld == 'org':
                credibility_score, source_type = 0.55, "organization"
                reasoning.append("Organization domain (.org)")
            elif tld in ['com', 'net']:
                credibility_score, source_type = 0.50, "commercial"
                reasoning.append("Commercial domain (needs verification)")
 
        content_boost = 0
        positive_patterns = [
            (r'according to (study|research|data|report|analysis)', 0.10),
            (r'published in (the journal|nature|science|the lancet)', 0.15),
            (r'(peer-reviewed|systematic review|meta-analysis)', 0.15),
            (r'official (statement|data|report|announcement)', 0.10),
            (r'(government|agency|ministry) (released|reported|announced)', 0.10),
            (r'(cites|references) (sources|studies|experts)', 0.08),
            (r'according to (CDC|WHO|FDA|NIH|Harvard|Oxford|Cambridge)', 0.12),
        ]
        negative_patterns = [
            (r"(you won't believe|shocking|amazing|incredible|mind-blowing)", -0.10),
            (r'(conspiracy|cover-up|they don\'t want you to know)', -0.15),
            (r'(click here|click now|limited time|secret)', -0.08),
            (r'(doctors hate this|experts don\'t want you to know)', -0.15),
            (r'(censored|banned|suppressed|hidden truth)', -0.12),
            (r'(miracle cure|guaranteed results|100% effective)', -0.12),
        ]
        for pattern, boost in positive_patterns:
            if re.search(pattern, snippet.lower()) or re.search(pattern, title.lower()):
                content_boost += boost
                reasoning.append("Found positive content indicator")
                break
        for pattern, penalty in negative_patterns:
            if re.search(pattern, snippet.lower()) or re.search(pattern, title.lower()):
                content_boost += penalty
                reasoning.append("Found concerning content indicator")
                break
 
        credibility_score = min(1.0, max(0.0, credibility_score + content_boost))
 
        if credibility_score >= 0.80:
            level = "VERY HIGH"
        elif credibility_score >= 0.65:
            level = "HIGH"
        elif credibility_score >= 0.50:
            level = "MEDIUM"
        elif credibility_score >= 0.35:
            level = "LOW"
        else:
            level = "VERY LOW"
 
        return {
            'url': url,
            'domain': domain,
            'credibility_score': round(credibility_score, 3),
            'credibility_level': level,
            'source_type': source_type,
            'reasoning': reasoning,
            'is_trusted': credibility_score >= 0.65,
        }
 
 
# ===========================================================================
# Misinformation Detection Agent
# ===========================================================================
class MisinformationDetectionAgent:
    """Detect misinformation by searching the web and analysing content using
    Groq LLM + Serper Google Search API with date awareness and source
    credibility assessment."""
 
    def __init__(self, groq_api_key: str, serper_api_key: str, model_name: str = "openai/gpt-oss-120b"):
        self.groq_client = Groq(api_key=groq_api_key)
        self.serper_api_key = serper_api_key
        self.model_name = model_name
        self.current_date = datetime.now()
        self.credibility_scorer = SourceCredibilityScorer()
        self.serper_headers = {
            'X-API-KEY': serper_api_key,
            'Content-Type': 'application/json',
        }
        self.serper_url = "https://google.serper.dev/search"
        self.news_url  = "https://google.serper.dev/news"
 
    def get_current_date_info(self) -> Dict:
        return {
            'full_date':      self.current_date.strftime("%B %d, %Y"),
            'year_month_day': self.current_date.strftime("%Y-%m-%d"),
            'month_day_year': self.current_date.strftime("%m/%d/%Y"),
            'day_month_year': self.current_date.strftime("%d/%m/%Y"),
            'weekday':        self.current_date.strftime("%A"),
            'month':          self.current_date.strftime("%B"),
            'year':           self.current_date.strftime("%Y"),
            'timestamp':      self.current_date.strftime("%Y%m%d_%H%M%S"),
        }
 
    def extract_date_from_text(self, text: str) -> str:
        date_patterns = [
            r'(today|yesterday|tomorrow)',
            r'(this week|last week|next week)',
            r'(this month|last month|next month)',
            r'(this year|last year|next year)',
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}',
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        return ""
 
    def search_web(self, query: str, num_results: int = 5, use_current_date: bool = True) -> List[Dict]:
        try:
            date_info = self.get_current_date_info()
            extracted_date = self.extract_date_from_text(query)
            enhanced_query = query
            if use_current_date:
                if extracted_date:
                    enhanced_query = f"{query} after:{date_info['year_month_day']}"
                elif date_info['year'] not in query:
                    enhanced_query = f"{query} {date_info['year']}"
 
            payload = json.dumps({
                "q": enhanced_query, "num": num_results,
                "gl": "us", "hl": "en",
            })
            news_payload = json.dumps({
                "q": enhanced_query, "num": num_results,
                "gl": "us", "hl": "en", "tbs": "qdr:d",
            })
 
            response      = requests.post(self.serper_url, headers=self.serper_headers, data=payload,      timeout=10)
            news_response = requests.post(self.news_url,   headers=self.serper_headers, data=news_payload, timeout=10)
 
            results: List[Dict] = []
 
            if response.status_code == 200:
                for item in response.json().get('organic', [])[:num_results]:
                    cred = self.credibility_scorer.assess_credibility(
                        item.get('link', '#'), item.get('snippet', ''), item.get('title', ''))
                    results.append({
                        'title': item.get('title', 'No title'),
                        'snippet': item.get('snippet', 'No description'),
                        'link': item.get('link', '#'),
                        'position': item.get('position', 0),
                        'source_type': 'web',
                        'credibility': cred,
                    })
 
            if news_response.status_code == 200:
                for item in news_response.json().get('news', [])[:num_results]:
                    cred = self.credibility_scorer.assess_credibility(
                        item.get('link', '#'), item.get('snippet', ''), item.get('title', ''))
                    results.append({
                        'title': item.get('title', 'No title'),
                        'snippet': item.get('snippet', 'No description'),
                        'link': item.get('link', '#'),
                        'position': len(results) + 1,
                        'source_type': 'news',
                        'date': item.get('date', 'Recent'),
                        'credibility': cred,
                    })
 
            results.sort(key=lambda x: x.get('credibility', {}).get('credibility_score', 0), reverse=True)
            return results[:num_results]
 
        except Exception as e:
            return []
 
    def search_current_news(self, topic: str, days_back: int = 7) -> List[Dict]:
        try:
            end_date   = self.current_date
            start_date = self.current_date - timedelta(days=days_back)
            news_payload = json.dumps({
                "q": topic, "num": 10, "gl": "us", "hl": "en",
                "tbs": f"cdr:1,cd_min:{start_date.strftime('%m/%d/%Y')},cd_max:{end_date.strftime('%m/%d/%Y')}",
            })
            response = requests.post(self.news_url, headers=self.serper_headers, data=news_payload, timeout=10)
            if response.status_code != 200:
                return []
            results: List[Dict] = []
            for item in response.json().get('news', [])[:10]:
                cred = self.credibility_scorer.assess_credibility(
                    item.get('link', '#'), item.get('snippet', ''), item.get('title', ''))
                results.append({
                    'title': item.get('title', 'No title'),
                    'snippet': item.get('snippet', 'No description'),
                    'link': item.get('link', '#'),
                    'date': item.get('date', 'Recent'),
                    'source': item.get('source', 'Unknown'),
                    'credibility': cred,
                })
            results.sort(key=lambda x: x.get('credibility', {}).get('credibility_score', 0), reverse=True)
            return results
        except Exception:
            return []
 
    def analyze_claim(self, claim: str, search_results: List[Dict]) -> Dict:
        if not search_results:
            return {'claim': claim, 'analysis': "No search results available.", 'search_results': []}
 
        date_info = self.get_current_date_info()
        extracted_date = self.extract_date_from_text(claim)
 
        total_cred  = sum(r.get('credibility', {}).get('credibility_score', 0) for r in search_results)
        avg_cred    = total_cred / len(search_results)
        high_cred   = [r for r in search_results if r.get('credibility', {}).get('credibility_score', 0) >= 0.65]
 
        context  = f"CURRENT DATE: {date_info['full_date']} ({date_info['weekday']})\n"
        context += "=" * 50 + "\n"
        context += f"OVERALL SOURCE CREDIBILITY: {avg_cred:.2%}\n"
        context += f"HIGH-CREDIBILITY SOURCES: {len(high_cred)} out of {len(search_results)}\n"
        if extracted_date:
            context += f"DATE REFERENCE IN CLAIM: {extracted_date}\n"
        context += "=" * 50 + "\n\nSEARCH RESULTS (SORTED BY CREDIBILITY):\n" + "=" * 50 + "\n"
 
        for i, result in enumerate(search_results, 1):
            cred = result.get('credibility', {})
            context += (
                f"\nSOURCE {i}:\n"
                f"Credibility Score: {cred.get('credibility_score', 0):.1%} ({cred.get('credibility_level', 'UNKNOWN')})\n"
                f"Source Type: {cred.get('source_type', 'unknown').upper()}\n"
                f"Domain: {cred.get('domain', 'unknown')}\n"
                f"Title: {result['title']}\n"
                f"Content: {result['snippet']}\n"
                f"URL: {result['link']}\n"
            )
            if 'date' in result:
                context += f"Publication Date: {result['date']}\n"
            context += "-" * 40 + "\n"
 
        prompt = f"""You are an expert fact-checker. Analyse the claim below based on the search results, paying attention to timing and source credibility.
 
CURRENT DATE: {date_info['full_date']}
{'DATE MENTIONED IN CLAIM: ' + extracted_date if extracted_date else ''}
OVERALL CREDIBILITY OF SOURCES: {avg_cred:.1%}
 
CLAIM TO VERIFY: "{claim}"
 
{context}
 
IMPORTANT CONSIDERATIONS:
1. Pay attention to whether the claim references current or historical information.
2. Check if the search results are timely and relevant to the current date.
3. GIVE SIGNIFICANT WEIGHT to sources with HIGH CREDIBILITY SCORES (>80%).
4. Sources with low credibility (<40%) should be treated skeptically.
5. Multiple high-credibility sources agreeing increases confidence.
 
Based on the search results, provide your analysis in EXACTLY this format:
 
VERDICT: [TRUE / FALSE / PARTIALLY TRUE / MISLEADING / UNVERIFIABLE / OUTDATED]
CONFIDENCE: [High/Medium/Low]
TIMELINESS: [Current/Recent/Outdated/Unclear]
CREDIBILITY_WEIGHT: [The verdict is primarily based on X high-credibility sources out of Y total sources]
EXPLANATION: (2-3 sentences explaining the reasoning, noting timing issues and which credible sources support the verdict)
EVIDENCE: (Key evidence from HIGH-CREDIBILITY sources, if available; otherwise note limitations)
 
Keep your response concise and easy to understand for a general audience."""
 
        try:
            completion = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": (
                        "You are an expert fact-checker and misinformation detection specialist "
                        "with a focus on temporal accuracy, current events, and source credibility. "
                        "Always prioritise high-credibility sources (.gov, .edu, major news) over "
                        "low-credibility sources (social media, blogs)."
                    )},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=800,
            )
            analysis = completion.choices[0].message.content
            return {
                'claim': claim,
                'analysis': analysis,
                'search_results': search_results,
                'analysis_date': date_info['full_date'],
                'aggregate_credibility': avg_cred,
                'high_cred_sources_count': len(high_cred),
            }
        except Exception as e:
            return {
                'claim': claim,
                'analysis': f"Error analyzing claim: {str(e)}",
                'search_results': search_results,
            }
 
    def detect_misinformation(self, claim: str, prioritize_news: bool = True) -> Dict:
        """Main entry point – search + deduplicate + analyse."""
        web_results  = self.search_web(claim, num_results=5, use_current_date=True)
        days_back    = 3 if prioritize_news else 7
        news_results = self.search_current_news(claim, days_back=days_back)
 
        all_results = web_results + news_results
        seen_urls: Dict[str, Dict] = {}
        for result in all_results:
            url = result.get('link')
            if not url:
                continue
            if url not in seen_urls:
                seen_urls[url] = result
            else:
                existing_cred = seen_urls[url].get('credibility', {}).get('credibility_score', 0)
                new_cred      = result.get('credibility', {}).get('credibility_score', 0)
                if new_cred > existing_cred:
                    seen_urls[url] = result
 
        unique_results = sorted(
            seen_urls.values(),
            key=lambda x: x.get('credibility', {}).get('credibility_score', 0),
            reverse=True,
        )
        search_results = list(unique_results)[:5]
 
        if not search_results:
            return {
                'claim': claim,
                'error': "No search results found. Please try a different query.",
                'search_results': [],
            }
 
        return self.analyze_claim(claim, search_results)
 
 
# ===========================================================================
# Pydantic models for API I/O
# ===========================================================================
class ClaimRequest(BaseModel):
    claim: str = Field(..., min_length=5, description="The claim text to fact-check")
    prioritize_news: bool = Field(True, description="If True, search news from last 3 days; otherwise last 7 days")
 
class CredibilityInfo(BaseModel):
    url: str
    domain: str
    credibility_score: float
    credibility_level: str
    source_type: str
    reasoning: List[str]
    is_trusted: bool
 
class SearchResult(BaseModel):
    title: str
    snippet: str
    link: str
    source_type: str
    credibility: CredibilityInfo
    date: Optional[str] = None
    source: Optional[str] = None
 
class VerificationResponse(BaseModel):
    claim: str
    analysis: str
    analysis_date: Optional[str] = None
    aggregate_credibility: Optional[float] = None
    high_cred_sources_count: Optional[int] = None
    search_results: List[SearchResult]
    error: Optional[str] = None