import streamlit as st
from PIL import Image
import pandas as pd
from financial_analysis import (
    generate_market_report_perplexity,
    generate_company_profile,
    upload_to_pinecone,
    get_industry_averages,
    compare_to_industry_average,
    analyze_company_standing
)
import os
import time

# Set page config
st.set_page_config(page_title="FinBot", layout="wide")

# ------------------------ CUSTOM STYLING ------------------------

primary_color = "#6bc72e"
text_color = "#333333"

st.markdown(f""" <style>
/* Global App Styles */
.stApp {{  
    background-color: #ffffff;  
    font-family: 'Arial', sans-serif;  
}}  
.stSidebar {{  
    background-color: {primary_color};  
    padding: 20px;  
}}  
.stSidebar h3, .stSidebar h4, .stSidebar .highlight {{  
    color: white;  
    font-size: 1.2em;  
    font-weight: bold;  
}}  
.stSidebar p {{  
    color: {text_color};  
}}  

.stButton > button {{  
    background-color: white;  
    color: {primary_color};  
    border: 2px solid {primary_color};  
    border-radius: 5px;  
    padding: 10px;  
    font-size: 16px;  
}}  
.stButton > button:hover, .stButton > button:focus, .stButton > button:active {{  
    background-color: {primary_color} !important;  
    color: white !important;  
    border: 2px solid {primary_color};  
    border-radius: 5px;  
    outline: none !important;  
    box-shadow: none !important;  
}}  

input, textarea, select {{  
    border: 1px solid #cccccc !important;  
    border-radius: 5px;  
    padding: 10px;  
    font-size: 16px;  
}}  

input:focus, textarea:focus, select:focus {{  
    outline: none !important;  
    border-radius: 5px;  
    padding: 10px;  
    font-size: 16px;  
}}  

.stTabs [data-baseweb="tab"] {{
    height: 45px;
    background-color: #ffffff;
    border-radius: 8px 8px 0px 0px;
    font-weight: bold;
    padding: 12px;
    color: {text_color} !important;
    border-bottom: none;
}}

.stTabs [aria-selected="true"] {{
    color: {primary_color} !important;
    border-bottom: 3px solid {primary_color} !important;
}}

h1, h2, h3 {{
    color: {primary_color} !important;
}}

.finbot-image {{
    display: flex;
    justify-content: flex-end;
    align-items: flex-end;
    height: 100%;
}}

.sidebar-title {{
    color: white !important;
}}

.market-report {{
    background-color: #f9f9f9;
    border-left: 5px solid {primary_color};
    padding: 20px;
    margin-top: 20px;
    border-radius: 5px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    font-family: 'Roboto', sans-serif;
}}
.market-report h4 {{
    color: #4a4a4a;
    margin-bottom: 15px;
    font-size: 24px;
    font-weight: 600;
}}
.market-report p {{
    line-height: 1.6;
    color: {text_color};
    font-size: 16px;
}}
</style>
""", unsafe_allow_html=True)

# ------------------------ SESSION STATE INIT ------------------------

if 'company_name' not in st.session_state:
    st.session_state.company_name = ""
if 'market_report' not in st.session_state:
    st.session_state.market_report = ""
if 'show_prompt' not in st.session_state:
    st.session_state.show_prompt = False
if 'company_profile' not in st.session_state:
    st.session_state.company_profile = None
if 'profile_generated' not in st.session_state:
    st.session_state.profile_generated = False

# ------------------------ SIDEBAR ------------------------

try:
    main_logo = Image.open("images/main_logo.png")
    overview_image = Image.open("images/overview.jpeg")
    finbot_image = Image.open("images/finbot.jpg")
except:
    main_logo = None
    overview_image = None
    finbot_image = None

with st.sidebar:
    if main_logo:
        st.image(main_logo, use_container_width=True)
    st.markdown("""
    <h3 class="sidebar-title">mypocketCFO: Your Financial Companion</h3>
    <p>Empowering your financial journey with real-time insights and AI-driven analysis.</p>
    <h4>Key Features:</h4>
    <ul>
      <li>Automated bookkeeping and reporting</li>
      <li>Cash flow forecasting and budgeting</li>
      <li>Personalized financial advice</li>
      <li>Integration with popular accounting systems</li>
    </ul>
    <p class="highlight">Join us on your journey to financial success!</p>
    """, unsafe_allow_html=True)

# ------------------------ TABS ------------------------

tab1, tab2, tab3 = st.tabs(['Overview', 'Profile', 'Analysis'])

# ------------------------ TAB 1: OVERVIEW ------------------------

with tab1:
    st.header("Overview", help="Market insights and basic company information")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Company Information")
        company_name = st.text_input("Company Name",
                                     value=st.session_state.company_name,
                                     key="company_name_input",
                                     label_visibility="collapsed")
        if st.button("Generate Market Report"):
            if not company_name:
                st.session_state.show_prompt = True
                st.session_state.company_name = ""
                st.session_state.market_report = ""
                st.session_state.profile_generated = False
            else:
                st.session_state.company_name = company_name
                with st.spinner("Generating market report..."):
                    market_report = generate_market_report_perplexity(company_name)
                    if market_report.startswith("Error:"):
                        st.error(market_report)
                        st.session_state.market_report = ""
                    else:
                        st.session_state.market_report = market_report
                        st.session_state.show_prompt = False

        if st.session_state.show_prompt:
            st.markdown('<p style="color:red;">Please enter a company name.</p>', unsafe_allow_html=True)

    with col2:
        if overview_image:
            st.image(overview_image, use_container_width=True)

    if st.session_state.market_report:
        st.markdown(f"""
        <div class="market-report">
        <h4>Market Report</h4>
        <p>{st.session_state.market_report}</p>
        </div>
        """, unsafe_allow_html=True)

# ------------------------ TAB 2: PROFILE ------------------------

with tab2:
    st.header("Company Profile", help="Detailed company and founder profile")

    if st.session_state.company_name:
        if st.button("Generate Comprehensive Profile") or st.session_state.profile_generated:
            if not st.session_state.profile_generated:
                with st.spinner("Generating company profile..."):
                    try:
                        profile_data = generate_company_profile(st.session_state.company_name)
                        st.session_state.company_profile = profile_data
                        st.session_state.profile_generated = True
                    except Exception as e:
                        st.error(f"Profile generation error: {str(e)}")

            if st.session_state.company_profile:
                profile_data = st.session_state.company_profile
                st.subheader(f"Profile: {st.session_state.company_name}")

                # Show source URLs if available
                with st.expander("✔ Data Sources"):
                    website = profile_data['source_urls'].get('website', "Not available")
                    founders = profile_data['source_urls'].get('founder_sources', [])
                    st.markdown(f"**Website:** {website}")
                    if founders:
                        st.markdown("**Founder Sources:**")
                        for link in founders:
                            st.markdown(f"- {link}")

                # Correctly split the profile into mini-tabs
                profile_subtab_titles = [
                    "Company Overview",
                    "Leadership Information",
                    "Founder Details",
                    "Financial Health",
                    "Business Operations",
                    "Mission and Values",
                    "Market Position"
                ]

                profile_subtab_contents = [
                    profile_data.get('company_overview', 'Not available'),
                    profile_data.get('leadership', 'Not available'),
                    profile_data.get('founder_details', 'Not available'),
                    profile_data.get('financial_health', 'Not available'),
                    profile_data.get('operations', 'Not available'),
                    profile_data.get('mission_values', 'Not available'),
                    profile_data.get('market_position', 'Not available')
                ]

                profile_subtabs = st.tabs(profile_subtab_titles)

                for idx, subtab in enumerate(profile_subtabs):
                    with subtab:
                        content = profile_subtab_contents[idx]
                        if content and content.strip().lower() != "not available":
                            for line in content.split('\n'):
                                if line.strip():
                                    st.markdown(f"- {line.strip()}")

                        # Special handling for Founder Literacy inside Founder Details tab
                        if profile_subtab_titles[idx] == "Founder Details":
                            founder_literacy = profile_data.get('financial_literacy', 'Unknown')
                            st.markdown(f"- **Founder Financial Literacy:** {founder_literacy}")

                # Downloadable Profile
                full_profile_text = f"""
                # {st.session_state.company_name} Profile

                ## Company Overview
                {profile_data.get('company_overview', '')}

                ## Leadership Information
                {profile_data.get('leadership', '')}

                ## Founder Details
                {profile_data.get('founder_details', '')}
                Financial Literacy: {profile_data.get('financial_literacy', '')}

                ## Financial Health
                {profile_data.get('financial_health', '')}

                ## Business Operations
                {profile_data.get('operations', '')}

                ## Mission and Values
                {profile_data.get('mission_values', '')}

                ## Market Position
                {profile_data.get('market_position', '')}
                """

                st.download_button(
                    label="Download Full Profile",
                    data=full_profile_text,
                    file_name=f"{st.session_state.company_name}_profile.md",
                    mime="text/markdown"
                )

        else:
            st.info("Please generate a Market Report in the Overview tab first.")
    else:
        st.info("Please enter a company name in the Overview tab first.")

# ------------------------ TAB 3: FINANCIAL ANALYSIS ------------------------

with tab3:
    col1, col2 = st.columns([3, 1])

    with col1:
        st.header("Financial Analysis", help="In-depth AI-driven financial and strategic evaluation")
        st.write("*Analyze your company's financial vitals and get personalized growth strategies.*")

        company_file = st.file_uploader("Upload your company's Income Statement CSV", type="csv")

        process_data = st.button("Process Data and Generate Analysis")

    with col2:
        if finbot_image:
            st.markdown('<div class="finbot-image">', unsafe_allow_html=True)
            st.image(finbot_image, width=500)
            st.markdown('</div>', unsafe_allow_html=True)

    if process_data:
        if not st.session_state.company_name:
            st.markdown('<p style="color:red;">Please enter a company name first.</p>', unsafe_allow_html=True)
        elif not company_file:
            st.markdown('<p style="color:red;">Please upload your company\'s Income Statement CSV file.</p>', unsafe_allow_html=True)
        else:
            with st.spinner("Processing uploaded data and generating detailed financial analysis..."):
                try:
                    # Upload benchmark files to Pinecone
                    industry_files = ["data/income_statement1.csv", "data/income_statement2.csv"]
                    upload_to_pinecone(industry_files)
                    time.sleep(2)

                    # Fetch industry averages
                    industry_averages = get_industry_averages()

                    if not industry_averages:
                        st.error("Failed to fetch industry benchmarks. Please retry.")
                    else:
                        # Compare uploaded company file
                        comparison_results = compare_to_industry_average(company_file, industry_averages)

                        st.subheader("Comparison to Industry Averages")
                        comparison_data = []
                        for metric, values in comparison_results.items():
                            comparison_data.append([
                                metric,
                                values['Company Value'],
                                values['Industry Average'],
                                values['Verdict']
                            ])
                        df_comparison = pd.DataFrame(comparison_data, columns=['Metric', 'Company Value', 'Industry Average', 'Verdict'])
                        st.dataframe(df_comparison)

                        # Generate full company analysis
                        st.subheader("Comprehensive Company Analysis")

                        company_statement = df_comparison.to_dict()

                        market_report = st.session_state.market_report
                        company_profile = st.session_state.company_profile if st.session_state.profile_generated else None

                        analysis = analyze_company_standing(
                            company_statement,
                            industry_averages,
                            market_report,
                            st.session_state.company_name,
                            company_profile
                        )

                        # Clean any stray tags if any
                        analysis = analysis.replace('<think>', '').replace('</think>', '')

                        st.markdown(analysis)

                except Exception as e:
                    st.error(f"Error processing financial analysis: {str(e)}")
