import pandas as pd
import numpy as np
import re
from groq import Groq
import requests
import os
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential
from pinecone import Pinecone, ServerlessSpec
import time
from serpapi import GoogleSearch
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import defaultdict
import json
from serpapi.google_search import GoogleSearch

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "financial-data-index"

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(index_name)

# Set up API keys
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
perplexity_api_key = os.environ.get("PERPLEXITY_API_KEY")
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Simple cache for market reports
market_report_cache = {}

def search_company_website(company_name):
    """Find official website and founder information using search"""
    try:
        # First search for official website
        website_params = {
            "q": f"{company_name} official website",
            "api_key": SERPAPI_API_KEY,
            "num": 3
        }

        website_search = GoogleSearch(website_params)
        website_results = website_search.get_dict()
        website_url = None

        if website_results.get("organic_results"):
            for result in website_results["organic_results"]:
                if company_name.lower() in result.get("link", "").lower():
                    website_url = result["link"]
                    break
            if not website_url:
                website_url = website_results["organic_results"][0]['link']

        # Then search for founder information
        founder_params = {
            "q": f"{company_name} founder OR founders OR co-founder OR co-founders",
            "api_key": SERPAPI_API_KEY,
            "num": 5
        }

        founder_search = GoogleSearch(founder_params)
        founder_results = founder_search.get_dict()
        founder_urls = []

        if founder_results.get("organic_results"):
            for result in founder_results["organic_results"]:
                url = result.get("link", "")
                if url and any(term in url.lower() for term in ['founder', 'about', 'team', 'leadership']):
                    founder_urls.append(url)

        return {
            'website_url': website_url,
            'founder_urls': founder_urls[:3] # Return top 3 most relevant URLs
        }

    except Exception as e:
        print(f"Search Error: {str(e)}")
        return {'website_url': None, 'founder_urls': []}

def scrape_website(url):
    """Enhanced website scraping with better content extraction"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'header', 'aside', 'form']):
            element.decompose()

        # Prioritize main content areas
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|main', re.IGNORECASE))

        if main_content:
            text = main_content.get_text()
        else:
            text = soup.get_text()

        # Clean up text
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:10000]  # First 10,000 characters for more context

    except Exception as e:
        print(f"Scraping Error: {str(e)}")
        return ""

def extract_founder_info(text, company_name):
    """Extract founder information from scraped text"""
    try:
        # Look for founder-related patterns
        patterns = [
            rf'(founder|co-founder|ceo|creator)\s*(?:of)?\s*({company_name})?\s*:\s*([^\n]+)',
            rf'({company_name})\s*(?:was)\s*(?:founded)\s*(?:by|in)\s*([^\n]+?)\s*(?:in|on|with)',
            rf'([A-Z][a-z]+ [A-Z][a-z]+)\s*(?:is|was)\s*(?:the|a)\s*(?:founder|co-founder|creator)'
        ]

        founders = set()
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.group(1) if 'group' in match.groupdict() else match.group(2)
                if name and len(name.split()) >= 2: # At least first and last name
                    founders.add(name.strip())

        return list(founders)

    except Exception as e:
        print(f"Founder extraction error: {str(e)}")
        return []

def analyze_financial_literacy(text):
    """Analyze financial sophistication using text"""
    try:
        scores = analyzer.polarity_scores(text)
        financial_terms = len(re.findall(r'\b(revenue|profit|margin|growth|financial|investment|capital|ROI|ROE|ROA|EBITDA)\b', text, re.IGNORECASE))

        if financial_terms > 10 and scores['compound'] >= 0.5:
            return "Highly Sophisticated"
        elif financial_terms > 5:
            return "Moderately Sophisticated"
        else:
            return "Basic Understanding"
    except:
        return "Unknown"

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def generate_company_profile(company_name):
    """Generate comprehensive company profile with enhanced founder information"""
    try:
        # Get website and founder URLs
        urls = search_company_website(company_name)
        website_url = urls['website_url']
        founder_urls = urls['founder_urls']
        
        # Scrape main website
        scraped_data = scrape_website(website_url) if website_url else "No website found"

        # Scrape founder information from multiple sources
        founder_data = []
        for url in founder_urls:
            founder_text = scrape_website(url)
            if founder_text:
                founder_data.append(founder_text)

        # Combine all data for processing
        combined_data = f"""  
        MAIN WEBSITE CONTENT:  
        {scraped_data}

        FOUNDER INFORMATION SOURCES:  
        {' '.join(founder_data)}  
        """

        prompt = f"""Please analyze this scraped company data and generate a detailed, well-structured profile for {company_name}:

        {combined_data}

        The profile should include these sections with rich, specific information:

        ## Company Overview  
        - Full legal name  
        - Year founded  
        - Headquarters location  
        - Industry sector  
        - Core business activities  
        - Major products/services  
        - Brief history (1 paragraph)  

        ## Leadership Information  
        ### Executive Team  
        - CEO: Name, tenure, background, education, notable achievements  
        - Other C-level executives: Names and brief profiles  
        - Board members: Names and affiliations if available  

        ### Founder Details (MUST INCLUDE THIS SECTION)  
        - Full names of all founders  
        - Detailed background for each founder including:  
          * Education history  
          * Previous professional experience  
          * Notable achievements  
          * Current roles besides this company  
          * Awards and recognition received  
        - Founding story (how and why the company was started)
        - Founder's current involvement with the company

        ### Financial Health Assessment
        - Revenue trends if mentioned
        - Profitability indicators
        - Growth metrics
        - Any available financial ratios
        - Funding history if applicable

        ### Business Operations
        - Key business metrics
        - Operational highlights
        - Geographic reach
        - Employee count if available
        - Major partnerships

        ### Mission and Values
        - Official mission statement
        - Core values
        - Corporate social responsibility initiatives
        - Sustainability practices

        ### Market Position
        - Competitive advantages
        - Market share indicators
        - Industry rankings if available
        - Recent awards/recognitions

        Format the output as clean, organized markdown with proper section headers. Include only information that can be reasonably inferred from the provided data. For missing information, clearly state "Information not available". 
        """

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a business intelligence specialist that extracts and structures detailed company information. Pay special attention to founder details and ensure they are comprehensive. Provide well-organized output with clear section headers and bullet points."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="deepseek-r1-distill-llama-70b", 
            temperature=0.2, 
            max_tokens=4000
        )

        profile_text = response.choices[0].message.content

        # Process the structured response into sections
        profile = {
            'company_overview': extract_section(profile_text, "Company Overview"),
            'leadership': extract_section(profile_text, "Leadership Information"),
            'founder_details': extract_section(profile_text, "Founder Details"),
            'financial_health': extract_section(profile_text, "Financial Health Assessment"),
            'operations': extract_section(profile_text, "Business Operations"),
            'mission_values': extract_section(profile_text, "Mission and Values"),
            'market_position': extract_section(profile_text, "Market Position"),
            'scraped_data': combined_data[:2000] + "..." if len(combined_data) > 2000 else combined_data,
            'financial_literacy': analyze_financial_literacy(combined_data),
            'source_urls': {
                'website': website_url,
                'founder_sources': founder_urls
            }
        }

        return profile

    except Exception as e:
        print(f"Profile generation error: {str(e)}")
        return {
            'company_overview': 'Error generating profile',
            'leadership': 'Not available',
            'founder_details': 'Not available',
            'financial_health': 'Not available',
            'operations': 'Not available',
            'mission_values': 'Not available',
            'market_position': 'Not available',
            'scraped_data': 'Error during data collection',
            'financial_literacy': 'Unknown',
            'source_urls': {}
        }

def extract_section(text, section_title):
    """Enhanced section extractor that handles nested subsections"""
    patterns = [
        rf'## {section_title}(.*?)(?=## \w+)',
        rf'{section_title}:(.*?)(?=\n\w+:|$)'
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            content = re.sub(r'\s*[-*]\s*', '', content, flags=re.MULTILINE)
            content = re.sub(r'\n{3,}', '\n\n', content)
            return content if content else "Information not available"

    return "Information not available"

def calculate_key_vitals(df):
    """Calculates key financial vitals based on available row names."""
    def get_row_total(row_name):
        if row_name in df.index:
            try:
                value = df.loc[row_name, 'Total']
                return float(value) if not pd.isna(value) else 0
            except (KeyError, ValueError):
                return 0
        return 0

    total_income = get_row_total('Total Income')
    total_cost_of_goods_sold = get_row_total('Total Cost Of Goods Sold')
    gross_profit = get_row_total('Gross Profit')
    total_expenses = get_row_total('Total Expenses')
    net_profit = get_row_total('Net Profit')
    selling_expenses = get_row_total('Advertising & Selling Expense')
    general_admin_expenses = get_row_total('General & Administrative expenses')

    # Calculate ratios with proper error handling
    def safe_divide(numerator, denominator):
        return numerator / denominator if denominator != 0 else 0

    gross_margin = safe_divide(gross_profit, total_income) * 100
    operating_expense_ratio = safe_divide(total_expenses, total_income) * 100
    net_profit_margin = safe_divide(net_profit, total_income) * 100
    cost_of_goods_sold_percent = safe_divide(total_cost_of_goods_sold, total_income) * 100
    selling_expense_ratio = safe_divide(selling_expenses, total_income) * 100
    ga_expense_ratio = safe_divide(general_admin_expenses, total_income) * 100

    return {
        "Gross Profit Margin": round(gross_margin, 2),
        "Operating Expense Ratio": round(operating_expense_ratio, 2),
        "Net Profit Margin": round(net_profit_margin, 2),
        "Cost of Goods Sold Ratio": round(cost_of_goods_sold_percent, 2),
        "Selling Expense Ratio": round(selling_expense_ratio, 2),
        "G&A Expense Ratio": round(ga_expense_ratio, 2),
    }

def embed_data(data):
    """Improved embedding function"""
    # In production, replace with actual embedding model
    return np.random.rand(768).tolist()

def upload_to_pinecone(file_paths):
    """Upload financial data to Pinecone with enhanced error handling"""
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            if 'Name' in df.columns:
                df = df.set_index('Name')
            df.index = df.index.astype(str).str.strip()

            key_vitals = calculate_key_vitals(df)
            if not key_vitals:
                print(f"Skipping {file_path} - no valid metrics calculated")
                continue

            vector = embed_data(list(key_vitals.values()))
            metadata = key_vitals
            record_id = os.path.basename(file_path)

            # Upsert to Pinecone
            index.upsert(
                vectors=[{
                    "id": record_id,
                    "values": vector,
                    "metadata": metadata
                }],
                namespace="financial_data"
            )
            print(f"Successfully uploaded {file_path} to Pinecone")

        except Exception as e:
            print(f"Error uploading {file_path}: {str(e)}")

def get_industry_averages():
    """Retrieve and calculate industry averages with proper error handling"""
    try:
        # List all vectors in the namespace
        stats = index.describe_index_stats()
        if stats.namespaces.get("financial_data", {}).get("vector_count", 0) == 0:
            print("No data found in Pinecone index")
            return {}

        # Query to get all vectors
        query_result = index.query(
            vector=[0]*768, # Dummy vector
            top_k=100,
            include_metadata=True,
            namespace="financial_data"
        )
        if not query_result.matches:
            print("No matches found in Pinecone query")
            return {}

        # Extract all metadata
        all_vitals = [match.metadata for match in query_result.matches if match.metadata]

        if not all_vitals:
            print("No valid metadata found in Pinecone results")
            return {}

        # Calculate averages
        industry_averages = {}

        metric_keys = [
            "Gross Profit Margin",
            "Operating Expense Ratio",
            "Net Profit Margin",
            "Cost of Goods Sold Ratio",
            "Selling Expense Ratio",
            "G&A Expense Ratio"
        ]
        for key in metric_keys:
            values = [float(vitals.get(key, 0)) for vitals in all_vitals if vitals.get(key) is not None]
            if values:
                industry_averages[key] = round(sum(values) / len(values), 2)
            else:
                industry_averages[key] = 0.0

        print("Calculated industry averages:", industry_averages)
        return industry_averages

    except Exception as e:
        print(f"Error retrieving industry averages: {str(e)}")
        return {}

def compare_to_industry_average(uploaded_file, industry_averages):
    """Compare company data to industry averages with enhanced validation"""
    try:
        df_company = pd.read_csv(uploaded_file)
        if 'Name' in df_company.columns:
            df_company = df_company.set_index('Name')
        df_company.index = df_company.index.astype(str).str.strip()

        company_vitals = calculate_key_vitals(df_company)
        if not company_vitals:
            raise ValueError("No valid metrics calculated for company")

        comparison_results = {}

        for metric, company_value in company_vitals.items():
            industry_value = industry_averages.get(metric, 0)

            if metric.endswith("Margin"):
                verdict = "Outperforming" if company_value > industry_value else "Underperforming"
            elif metric.endswith("Ratio"):
                verdict = "Outperforming" if company_value < industry_value else "Underperforming"
            else:
                verdict = "N/A"

            comparison_results[metric] = {
                "Company Value": round(company_value, 2),
                "Industry Average": round(industry_value, 2),
                "Verdict": verdict,
                "Difference": round(company_value - industry_value, 2)
            }

        return comparison_results

    except Exception as e:
        print(f"Comparison error: {str(e)}")
        return {}

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def generate_market_report_perplexity(company_name):
    """Generate market report with enhanced error handling"""
    if company_name in market_report_cache:
        return market_report_cache[company_name]

    if not perplexity_api_key:
        return "Error: Perplexity API key is missing."

    url = "https://api.perplexity.ai/chat/completions"
    prompt = f"""Generate a detailed market report for {company_name} including:
        1. Company Overview
        2. Market Size and Growth
        3. Target Market
        4. Competitive Landscape
        5. Key Trends
        6. SWOT Analysis
        7. Future Outlook
        Use specific numbers and cite sources when possible."""

    payload = {
        "model": "sonar-pro",
        "messages": [
            {
                "role": "system",
                "content": "You are a financial analyst providing accurate, data-driven insights."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 3000,
        "temperature": 0.1
    }

    headers = {
        "Authorization": f"Bearer {perplexity_api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"]
        # Clean up response
        result = re.sub(r'\[\d+\]', '', result) # Remove citations
        result = re.sub(r'\n{3,}', '\n\n', result) # Remove excessive newlines
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL) # Remove AI thinking

        market_report_cache[company_name] = result
        return result

    except Exception as err:
        return f"Error generating report: {str(err)}"

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def analyze_company_standing(company_statement, industry_averages, market_report, company_name, company_profile=None):
    """Generate company analysis with enhanced structure and profile integration"""
    profile_context = ""
    if company_profile:
        profile_context = f"""
        Additional Company Context:
        - Company Overview: {company_profile.get('company_overview', 'N/A')}
        - Leadership: {company_profile.get('leadership', 'N/A')}
        - Founder Details: {company_profile.get('founder_details', 'N/A')}
        - Financial Health: {company_profile.get('financial_health', 'N/A')}
        - Operations: {company_profile.get('operations', 'N/A')}
        - Mission & Values: {company_profile.get('mission_values', 'N/A')}
        - Market Position: {company_profile.get('market_position', 'N/A')}
        """

    prompt = f"""Analyze {company_name}'s performance compared to industry benchmarks:

    Company Financial Data: {json.dumps(company_statement, indent=2)}
    Industry Averages: {json.dumps(industry_averages, indent=2)}
    Market Context: {market_report[:2000]}...
    {profile_context}

    Provide analysis in this structure:
    ### Financial Performance Summary
    (Brief overview of key findings, incorporating company background from profile if available)

    ### Strengths
    (3-5 specific strengths with data points, considering:
     - Financial metrics outperforming industry
     - Leadership/founder strengths from profile
     - Operational advantages mentioned in profile
     - Market position advantages)

    ### Weaknesses
    (3-5 specific weaknesses with data points, considering:
     - Financial metrics underperforming industry
     - Leadership/founder gaps from profile
     - Operational challenges mentioned in profile
     - Market position vulnerabilities)

    ### Recommendations
    (3-5 actionable recommendations tailored to:
     - Financial improvement opportunities
     - Leadership/management considerations from profile
     - Operational optimizations suggested by profile
     - Market positioning strategies)

    ### Overall Verdict
    (Comprehensive conclusion: Outperforming/Underperforming/On Par, considering both financials and qualitative profile factors)

    Use precise numbers and maintain consistency with the provided data. Incorporate relevant details from the company profile to make the analysis more personalized and insightful."""

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a financial analyst providing detailed, data-driven company analysis. Incorporate both quantitative financial data and qualitative profile information to create a comprehensive assessment. Provide clean output without any internal thinking tags."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="deepseek-r1-distill-llama-70b",
        temperature=0.1,
        max_tokens=3000
    )

    analysis = response.choices[0].message.content
    # Remove any remaining AI thinking patterns
    analysis = re.sub(r'<think>.*?</think>', '', analysis, flags=re.DOTALL)
    analysis = re.sub(r'\[.*?\]', '', analysis)
    return analysis