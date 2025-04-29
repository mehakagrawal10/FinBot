import streamlit as st
from PIL import Image
import pandas as pd
from financial_analysis import (
    generate_market_report_perplexity,
    compare_to_industry_average,
    calculate_key_vitals,
    upload_to_pinecone,
    get_industry_averages,
    analyze_company_standing,
    generate_company_profile
)
import os
from io import StringIO
import time

# Set page config
st.set_page_config(page_title="FinBot", layout="wide")

# Define colors
primary_color = "#6bc72e"
text_color = "#333333"

# Custom CSS
st.markdown(f""" <style>
/* Global Styles */
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

/* Button Styles */  
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

/* Input Fields without green borders */  
input, textarea, select {{  
    border: 1px solid #cccccc !important;  
    border-radius: 5px;  
    padding: 10px;  
    font-size: 16px;  
}}  

/* Remove focus shadow */  
input:focus, textarea:focus, select:focus {{  
    outline: none !important;  
    border-radius: 5px;  
    padding: 10px;  
    font-size: 16px;  
}}  
box-shadow: none !important;
}}

/* Expander Styles */
.stExpander {{
    border: 1px solid #d3d3d3;
}}
.stExpander:hover, .stExpander:focus, details[open] summary {{
    border-color: {primary_color} !important;
    color: {primary_color} !important;
    outline: none !important;
}}
details summary:hover {{
    color: {primary_color} !important;
}}

/* Navigation Tabs with single green underline */
.stTabs [data-baseweb="tab"] {{
    height: 45px;
    background-color: #ffffff;
    border-radius: 8px 8px 0px 0px;
    font-weight: bold;
    padding: 12px;
    color: {text_color} !important;
    border-bottom: none;
}}

/* Active tab: Force single green underline */
.stTabs [aria-selected="true"] {{
    color: {primary_color} !important;
    border-bottom: 3px solid {primary_color} !important;
}}

/* Ensure no red focus outline or shadow */
*:focus, *:active, *:hover {{
    outline: none !important;
    box-shadow: none !important;
}}

/* Override any remaining red borders */
[data-baseweb="tab-highlight"] {{
    background-color: transparent !important;
}}

/* Green headings */
h1, h2, h3 {{
    color: {primary_color} !important;
}}

/* FinBot image alignment */
.finbot-image {{
    display: flex;
    justify-content: flex-end;
    align-items: flex-end;
    height: 100%;
}}

/* White text for sidebar title */
.sidebar-title {{
    color: white !important;
}}

/* Market Report Styling */
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

# Initialize session state
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

# Load images
try:
    main_logo = Image.open("images/main_logo.png")
    overview_image = Image.open("images/overview.jpeg")
    finbot_image = Image.open("images/finbot.jpg")
except:
    # Fallback if images aren't found
    main_logo = None
    overview_image = None
    finbot_image = None

# Sidebar with branding and description
with st.sidebar:
    if main_logo:
        st.image(main_logo, use_container_width=True)
    st.markdown("""
    <h3 class="sidebar-title">mypocketCFO: Your Financial Companion</h3>

    mypocketCFO is an innovative financial management tool designed to empower small businesses and entrepreneurs. Our AI-driven platform provides real-time financial insights, predictive analytics, and personalized recommendations to help you make informed decisions and drive growth.

    <h4>Key Features:</h4>

    - Automated bookkeeping and financial reporting
    - Cash flow forecasting and budget optimization
    - Customized financial advice and strategy planning
    - Integration with popular accounting software

    <p class="highlight">Join us on your journey to success!</p>
    """, unsafe_allow_html=True)

# Tabs for navigation - Reordered as requested (Overview, Profile, Analysis)
tab1, tab2, tab3 = st.tabs(['Overview', "Profile", "Analysis"])

with tab1:
    st.header("Overview", help="Company overview and market insights")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Company Information")
        st.write("Enter the company name")
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
            st.markdown('<p class="prompt-box">Please enter a company name</p>', unsafe_allow_html=True)

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

with tab2:
    st.header("Company Profile", help="Detailed company and founder insights")

    if st.session_state.company_name:
        if st.button("Generate Comprehensive Profile") or st.session_state.profile_generated:
            if not st.session_state.profile_generated:
                with st.spinner("Generating detailed company profile..."):
                    try:
                        profile_data = generate_company_profile(st.session_state.company_name)
                        st.session_state.company_profile = profile_data
                        st.session_state.profile_generated = True
                    except Exception as e:
                        st.error(f"Profile generation error: {str(e)}")

            if st.session_state.company_profile:
                profile_data = st.session_state.company_profile
                st.subheader(f"Comprehensive Profile: {st.session_state.company_name}")

                # Source information
                with st.expander("✔ Data Source Information", expanded=False):
                    st.write("**Main Website:**")
                    st.write(profile_data['source_urls'].get('website', 'Not available'))

                    st.write("**Founder Information Sources:**")
                    for url in profile_data['source_urls'].get('founder_sources', []):
                        st.write(url)

                    st.write("**Raw Scraped Data Sample:**")
                    st.code(profile_data.get('scraped_data', 'No data available'), language='text')

                    st.write(f"**Content Sophistication:** {profile_data.get('financial_literacy', 'Unknown')}")

                # Main profile sections with added founder section
                with st.expander("📝 Company Overview", expanded=True):
                    st.markdown(profile_data.get('company_overview', 'Information not available'))

                with st.expander("📝 Leadership & Management"):
                    st.markdown(profile_data.get('leadership', 'Information not available'))

                with st.expander("📝 Founder Details", expanded=True):
                    founder_content = profile_data.get('founder_details', 'Information not available')
                    if founder_content == 'Information not available':
                        st.warning("Could not extract detailed founder information. Trying alternative methods...")
                    st.markdown(founder_content)

                with st.expander("💰 Financial Health"):
                    st.markdown(profile_data.get('financial_health', 'Information not available'))

                with st.expander("🏭 Business Operations"):
                    st.markdown(profile_data.get('operations', 'Information not available'))

                with st.expander("🌟 Mission & Values"):
                    st.markdown(profile_data.get('mission_values', 'Information not available'))

                with st.expander("📊 Market Position"):
                    st.markdown(profile_data.get('market_position', 'Information not available'))

                # Download button with founder info included
                profile_text = f""" 
                # {st.session_state.company_name} Profile Report

                ## Company Overview
                {profile_data.get('company_overview', '')}

                ## Leadership & Management
                {profile_data.get('leadership', '')}

                ## Founder Details
                {profile_data.get('founder_details', '')}

                ## Financial Health
                {profile_data.get('financial_health', '')}

                ## Business Operations
                {profile_data.get('operations', '')}

                ## Mission & Values
                {profile_data.get('mission_values', '')}

                ## Market Position
                {profile_data.get('market_position', '')}
                """

                st.download_button(
                    label="Download Full Profile",
                    data=profile_text,
                    file_name=f"{st.session_state.company_name}_profile.md",
                    mime="text/markdown"
                )
        else:
            st.info("Please generate a market report in the Overview tab first")
    else:
        st.info("Please enter a company name in the Overview tab to generate profile")

with tab3:
    col1, col2 = st.columns([3, 1])

    with col1:
        st.header("Financial Analysis", help="AI-powered financial analysis")
        st.write("*FinBot provides a deep dive into financial performance, identifying key trends and risks.*")

        # File uploader for single company CSV file
        company_file = st.file_uploader("Upload your company's CSV file", type="csv")

        process_data = st.button("Process Data and Generate Analysis")

    with col2:
        if finbot_image:
            st.markdown('<div class="finbot-image">', unsafe_allow_html=True)
            st.image(finbot_image, width=500)
            st.markdown('</div>', unsafe_allow_html=True)

    if process_data:
        if not st.session_state.company_name:
            st.markdown('<p class="prompt-box">Enter a company name in Overview first.</p>', unsafe_allow_html=True)
        elif not company_file:
            st.markdown('<p class="prompt-box">Please upload your company\'s CSV file to analyze.</p>', unsafe_allow_html=True)
        else:
            with st.spinner("Processing data and generating analysis..."):
                # Upload industry data to Pinecone
                industry_files = ["data/income_statement1.csv", "data/income_statement2.csv"]
                upload_to_pinecone(industry_files)

                # Add a delay to ensure data is processed
                time.sleep(2)

                # Get industry averages from Pinecone
                industry_averages = get_industry_averages()

                if not industry_averages:
                    st.error("Failed to retrieve industry data. Please try again.")
                else:
                    # Perform the comparison
                    try:
                        comparison_results = compare_to_industry_average(company_file, industry_averages)

                        st.subheader("Comparison to Industry Averages")

                        # Display the comparison results in a table
                        data = []
                        for metric, values in comparison_results.items():
                            data.append([metric, values['Company Value'], 
                                       values['Industry Average'], values['Verdict']])

                        df = pd.DataFrame(data, columns=['Metric', 'Company Value', 'Industry Average', 'Verdict'])
                        st.dataframe(df)

                        # Generate and display the company standing analysis
                        st.subheader("Comprehensive Company Analysis")
                        company_statement = df.to_dict()
                        market_report = st.session_state.market_report
                        
                        # Pass the company profile if available
                        company_profile = st.session_state.company_profile if st.session_state.profile_generated else None
                        
                        analysis = analyze_company_standing(
                            company_statement,
                            industry_averages,
                            market_report,
                            st.session_state.company_name,
                            company_profile
                        )
                        st.markdown(analysis)

                    except Exception as e:
                        st.error(f"Error processing file: {e}")
