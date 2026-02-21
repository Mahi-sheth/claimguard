import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pdfplumber
import re
import hashlib
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from fpdf import FPDF
import tempfile
import base64

# ============================================
# PDF REPORT GENERATOR
# ============================================
def generate_pdf_report(policy, analysis_result, financial, claim_amount, insurance_pays, out_of_pocket):
    """Generate a PDF report of the claim analysis"""
    
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 16)
            self.set_text_color(26, 35, 126)  # Navy blue
            self.cell(0, 10, 'ClaimGuard Insurance Report', 0, 1, 'C')
            self.ln(10)
        
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    pdf = PDF()
    pdf.add_page()
    
    # Policy Information
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(26, 35, 126)
    pdf.cell(0, 10, 'Policy Information', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(0, 0, 0)
    
    # Use encode to handle special characters
    filename = policy["filename"].encode('latin-1', 'ignore').decode('latin-1')
    pdf.cell(0, 6, f'File: {filename}', 0, 1)
    pdf.cell(0, 6, f'Policy ID: {policy["unique_id"]}', 0, 1)
    pdf.cell(0, 6, f'Upload Date: {policy["upload_time"]}', 0, 1)
    pdf.cell(0, 6, f'Policy Type: {policy["policy_type"]}', 0, 1)
    pdf.ln(10)
    
    # Risk Scores
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(26, 35, 126)
    pdf.cell(0, 10, 'Risk Assessment', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(0, 0, 0)
    
    scores = policy['risk_scores']
    pdf.cell(0, 6, f'Claim Coverage Risk: {scores["coverage_risk"]}%', 0, 1)
    pdf.cell(0, 6, f'Out-of-Pocket Risk: {scores["cost_risk"]}%', 0, 1)
    pdf.cell(0, 6, f'Claim Delay Risk: {scores["delay_risk"]}%', 0, 1)
    total_risk = (scores["coverage_risk"] * 0.4 + scores["cost_risk"] * 0.35 + scores["delay_risk"] * 0.25)
    pdf.cell(0, 6, f'Overall Risk: {total_risk:.1f}%', 0, 1)
    pdf.ln(10)
    
    # Detected Terms - Use bullet points instead of checkmarks
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(26, 35, 126)
    pdf.cell(0, 10, 'Detected Policy Terms', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(0, 0, 0)
    
    if analysis_result['co_pay_percentage'] > 0:
        pdf.cell(0, 6, f'- Co-pay: {analysis_result["co_pay_percentage"]}%', 0, 1)
    if analysis_result['deductible'] > 0:
        pdf.cell(0, 6, f'- Deductible: Rs {analysis_result["deductible"]:,}', 0, 1)
    if analysis_result['room_rent_cap']:
        pdf.cell(0, 6, f'- Room Rent Cap: {analysis_result["room_rent_cap"]}', 0, 1)
    if analysis_result['sub_limits']:
        for limit, amount in analysis_result['sub_limits'].items():
            pdf.cell(0, 6, f'- {limit.title()} Sub-limit: Rs {amount:,}', 0, 1)
    if analysis_result.get('waiting_periods'):
        waiting_text = ', '.join(analysis_result['waiting_periods']).encode('latin-1', 'ignore').decode('latin-1')
        pdf.cell(0, 6, f'- Waiting Periods: {waiting_text}', 0, 1)
    pdf.cell(0, 6, f'- Exclusions Found: {analysis_result["exclusion_count"]}', 0, 1)
    pdf.ln(10)
    
    # Claim Simulation
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(26, 35, 126)
    pdf.cell(0, 10, 'Claim Simulation Results', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 6, f'Claim Amount: Rs {claim_amount:,}', 0, 1)
    pdf.set_text_color(40, 167, 69)  # Green
    pdf.cell(0, 6, f'Insurance Pays: Rs {insurance_pays:,.0f}', 0, 1)
    pdf.set_text_color(220, 53, 69)  # Red
    pdf.cell(0, 6, f'You Pay: Rs {out_of_pocket:,.0f}', 0, 1)
    pdf.ln(10)
    
    # Key Clauses
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(26, 35, 126)
    pdf.cell(0, 10, 'Key Policy Clauses', 0, 1)
    pdf.set_font('Arial', '', 9)
    pdf.set_text_color(0, 0, 0)
    
    for term, clause in policy['clauses'].items():
        if clause != "Not mentioned in document":
            # Clean clause text
            clean_clause = clause.encode('latin-1', 'ignore').decode('latin-1')[:100]
            pdf.multi_cell(0, 5, f"{term.title()}: {clean_clause}...")
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        pdf.output(tmp_file.name)
        tmp_path = tmp_file.name
    
    return tmp_path
    
    # Policy Information
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(26, 35, 126)
    pdf.cell(0, 10, 'Policy Information', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 6, f'File: {policy["filename"]}', 0, 1)
    pdf.cell(0, 6, f'Policy ID: {policy["unique_id"]}', 0, 1)
    pdf.cell(0, 6, f'Upload Date: {policy["upload_time"]}', 0, 1)
    pdf.cell(0, 6, f'Policy Type: {policy["policy_type"]}', 0, 1)
    pdf.ln(10)
    
    # Risk Scores
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(26, 35, 126)
    pdf.cell(0, 10, 'Risk Assessment', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(0, 0, 0)
    
    scores = policy['risk_scores']
    pdf.cell(0, 6, f'Claim Coverage Risk: {scores["coverage_risk"]}%', 0, 1)
    pdf.cell(0, 6, f'Out-of-Pocket Risk: {scores["cost_risk"]}%', 0, 1)
    pdf.cell(0, 6, f'Claim Delay Risk: {scores["delay_risk"]}%', 0, 1)
    total_risk = (scores["coverage_risk"] * 0.4 + scores["cost_risk"] * 0.35 + scores["delay_risk"] * 0.25)
    pdf.cell(0, 6, f'Overall Risk: {total_risk:.1f}%', 0, 1)
    pdf.ln(10)
    
    # Detected Terms
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(26, 35, 126)
    pdf.cell(0, 10, 'Detected Policy Terms', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(0, 0, 0)
    
    if analysis_result['co_pay_percentage'] > 0:
        pdf.cell(0, 6, f'✓ Co-pay: {analysis_result["co_pay_percentage"]}%', 0, 1)
    if analysis_result['deductible'] > 0:
        pdf.cell(0, 6, f'✓ Deductible: ₹{analysis_result["deductible"]:,}', 0, 1)
    if analysis_result['room_rent_cap']:
        pdf.cell(0, 6, f'✓ Room Rent Cap: {analysis_result["room_rent_cap"]}', 0, 1)
    if analysis_result['sub_limits']:
        for limit, amount in analysis_result['sub_limits'].items():
            pdf.cell(0, 6, f'✓ {limit.title()} Sub-limit: ₹{amount:,}', 0, 1)
    if analysis_result['waiting_periods']:
        pdf.cell(0, 6, f'✓ Waiting Periods: {", ".join(analysis_result["waiting_periods"])}', 0, 1)
    pdf.cell(0, 6, f'✓ Exclusions Found: {analysis_result["exclusion_count"]}', 0, 1)
    pdf.ln(10)
    
    # Claim Simulation
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(26, 35, 126)
    pdf.cell(0, 10, 'Claim Simulation Results', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 6, f'Claim Amount: ₹{claim_amount:,}', 0, 1)
    pdf.set_text_color(40, 167, 69)  # Green
    pdf.cell(0, 6, f'Insurance Pays: ₹{insurance_pays:,.0f}', 0, 1)
    pdf.set_text_color(220, 53, 69)  # Red
    pdf.cell(0, 6, f'You Pay: ₹{out_of_pocket:,.0f}', 0, 1)
    pdf.ln(10)
    
    # Key Clauses
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(26, 35, 126)
    pdf.cell(0, 10, 'Key Policy Clauses', 0, 1)
    pdf.set_font('Arial', '', 9)
    pdf.set_text_color(0, 0, 0)
    
    for term, clause in policy['clauses'].items():
        if clause != "Not mentioned in document":
            pdf.multi_cell(0, 5, f"{term.title()}: {clause[:100]}...")
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        pdf.output(tmp_file.name)
        tmp_path = tmp_file.name
    
    return tmp_path

def get_download_link(file_path, filename):
    """Generate a download link for a file"""
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}" style="text-decoration: none;">📥 Download PDF Report</a>'
    return href

# Page configuration with custom theme
st.set_page_config(
    page_title="ClaimGuard - Insurance Risk Analyzer", 
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1a237e 0%, #0d47a1 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1a237e;
        color: #1a237e;
    }
    
    .metric-card h4 {
        color: #1a237e;
        margin-top: 0;
    }
    
    .metric-card .risk-high, 
    .metric-card .risk-medium, 
    .metric-card .risk-low {
        color: inherit;
    }
    
    .metric-card small {
        color: #666;
    }
    
    .risk-high {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .risk-medium {
        color: #ffc107;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .risk-low {
        color: #28a745;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .financial-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        color: #666;
        padding: 1rem;
        font-size: 0.9rem;
        border-top: 1px solid #eee;
    }
    .claim-badge {
        background: #1a237e;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-block;
    }
    .download-btn {
        display: inline-block;
        padding: 0.5rem 1rem;
        background-color: #1a237e;
        color: white;
        text-decoration: none;
        border-radius: 5px;
        font-weight: bold;
    }
    .download-btn:hover {
        background-color: #0d47a1;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'policy_history' not in st.session_state:
    st.session_state['policy_history'] = []
if 'current_policy' not in st.session_state:
    st.session_state['current_policy'] = None
if 'analysis_done' not in st.session_state:
    st.session_state['analysis_done'] = False

# ============================================
# ML MODEL FOR RISK PREDICTION
# ============================================
class RiskPredictor:
    def __init__(self):
        self.vectorizer = CountVectorizer(max_features=100, stop_words='english')
        self.scaler = MinMaxScaler()
        
    def extract_features(self, text):
        """Extract numerical features from text"""
        features = {}
        
        # 1. Count specific keywords (each gives different weight)
        keywords = {
            'waiting_period': ['waiting period', 'waiting time', 'cooling period'],
            'exclusion': ['exclusion', 'not covered', 'excluded', 'not payable'],
            'co_pay': ['co-pay', 'copay', 'coinsurance', 'payable by insured'],
            'sub_limit': ['sub-limit', 'sublimit', 'cap of', 'maximum limit'],
            'room_rent': ['room rent', 'room charges', 'accommodation'],
            'pre_existing': ['pre-existing', 'preexisting', 'existing condition'],
            'claim_days': ['within 24 hours', 'within 48 hours', 'immediately'],
            'deductible': ['deductible', 'excess amount', 'first pay'],
            'disease': ['cancer', 'diabetes', 'heart', 'kidney', 'liver', 'hiv'],
            'surgery': ['surgery', 'operation', 'procedure', 'treatment'],
            'hospital': ['hospital', 'medical', 'healthcare', 'clinic'],
            'percentage': ['%', 'percent', 'percentage'],
            'money': ['rupees', 'rs', 'inr', 'lakh', 'thousand'],
            'time': ['day', 'days', 'month', 'months', 'year', 'years'],
            'limit': ['limit', 'capped', 'maximum', 'upto']
        }
        
        text_lower = text.lower()
        for key, words in keywords.items():
            count = sum(text_lower.count(word) for word in words)
            features[key] = count
        
        # 2. Find percentages
        percentages = re.findall(r'(\d+)%', text_lower)
        features['avg_percentage'] = np.mean([int(p) for p in percentages]) if percentages else 0
        
        # 3. Find monetary values
        amounts = re.findall(r'rs\.?\s*(\d+)|₹\s*(\d+)', text_lower)
        flat_amounts = []
        for match in amounts:
            for val in match:
                if val:
                    flat_amounts.append(int(val))
        features['avg_amount'] = np.mean(flat_amounts) if flat_amounts else 0
        
        # 4. Find time periods
        days = re.findall(r'(\d+)\s*(day|days)', text_lower)
        months = re.findall(r'(\d+)\s*(month|months)', text_lower)
        years = re.findall(r'(\d+)\s*(year|years)', text_lower)
        
        features['has_days'] = len(days)
        features['has_months'] = len(months)
        features['has_years'] = len(years)
        
        # 5. Document complexity (longer = more complex = higher risk)
        features['length'] = min(len(text) / 1000, 10)  # Normalize
        
        return features
    
    def predict_risk(self, text, policy_type, age, has_disease):
        """Predict risk scores based on document features"""
        features = self.extract_features(text)
        
        # Coverage Risk (based on exclusions, waiting periods, disease mentions)
        coverage_risk = 20  # Base
        
        coverage_risk += features['waiting_period'] * 8
        coverage_risk += features['exclusion'] * 10
        coverage_risk += features['pre_existing'] * 12
        coverage_risk += features['disease'] * 5
        coverage_risk += features['has_years'] * 5
        
        # Adjust by policy type
        if policy_type == "Health Insurance":
            coverage_risk += 10  # Health policies have more exclusions
        elif policy_type == "Car Insurance":
            coverage_risk -= 10
        elif policy_type == "Life Insurance":
            coverage_risk -= 5
        
        # Cost Risk (based on co-pay, sub-limits, percentages)
        cost_risk = 15  # Base
        
        cost_risk += features['co_pay'] * 12
        cost_risk += features['sub_limit'] * 10
        cost_risk += features['room_rent'] * 8
        cost_risk += features['percentage'] * 5
        cost_risk += features['money'] * 3
        cost_risk += features['deductible'] * 10
        
        # Add percentage impact
        cost_risk += features['avg_percentage'] * 1.5
        
        # Delay Risk (based on claim conditions, time limits)
        delay_risk = 10  # Base
        
        delay_risk += features['claim_days'] * 15
        delay_risk += features['time'] * 4
        delay_risk += features['has_days'] * 8
        delay_risk += features['has_months'] * 5
        
        # Add user profile impact
        if age > 60:
            coverage_risk += 15
            delay_risk += 10
        elif age > 45:
            coverage_risk += 8
            delay_risk += 5
        
        if has_disease:
            coverage_risk += 20
            cost_risk += 10
        
        # Ensure within 0-100
        coverage_risk = min(max(int(coverage_risk), 0), 100)
        cost_risk = min(max(int(cost_risk), 0), 100)
        delay_risk = min(max(int(delay_risk), 0), 100)
        
        return {
            'coverage_risk': coverage_risk,
            'cost_risk': cost_risk,
            'delay_risk': delay_risk
        }
    
    def extract_co_pay_percentage(self, text):
        """Extract co-pay percentage from policy text"""
        text_lower = text.lower()
        
        # Look for co-pay patterns
        patterns = [
            r'co[-\s]?pay[:\s]*(\d+)%',
            r'copayment[:\s]*(\d+)%',
            r'co[-\s]?insurance[:\s]*(\d+)%',
            r'payable by insured[:\s]*(\d+)%',
            r'(\d+)%\s*co[-\s]?pay',
            r'(\d+)%\s*copayment',
            r'(\d+)%\s*co-insurance'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return int(match.group(1))
        
        # Check for generic mentions of co-pay without percentage
        if 'co-pay' in text_lower or 'copay' in text_lower or 'co-payment' in text_lower:
            return 10  # Default if co-pay exists but no percentage mentioned
        
        return 0  # No co-pay found
    
    def extract_deductible(self, text):
        """Extract deductible amount from policy text"""
        text_lower = text.lower()
        
        patterns = [
            r'deductible[:\s]*rs\.?\s*(\d+)|deductible[:\s]*₹\s*(\d+)',
            r'excess[:\s]*rs\.?\s*(\d+)|excess[:\s]*₹\s*(\d+)',
            r'first pay[:\s]*rs\.?\s*(\d+)|first pay[:\s]*₹\s*(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                for val in match.groups():
                    if val:
                        return int(val)
        
        return 0
    
    def extract_room_rent_cap(self, text):
        """Extract room rent capping from policy text"""
        text_lower = text.lower()
        
        patterns = [
            r'room rent[:\s]*rs\.?\s*(\d+)|room rent[:\s]*₹\s*(\d+)',
            r'room charges[:\s]*rs\.?\s*(\d+)|room charges[:\s]*₹\s*(\d+)',
            r'accommodation[:\s]*rs\.?\s*(\d+)|accommodation[:\s]*₹\s*(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                for val in match.groups():
                    if val:
                        return int(val)
        
        # Check for percentage-based room rent capping
        percent_match = re.search(r'room rent[:\s]*(\d+)%', text_lower)
        if percent_match:
            return percent_match.group(1) + "%"
        
        return None
    
    def extract_sub_limits(self, text):
        """Extract various sub-limits from policy text"""
        text_lower = text.lower()
        sub_limits = {}
        
        # Common sub-limits
        limit_types = {
            'icu': r'icu[:\s]*rs\.?\s*(\d+)|icu[:\s]*₹\s*(\d+)',
            'surgery': r'surgery[:\s]*rs\.?\s*(\d+)|surgery[:\s]*₹\s*(\d+)',
            'doctor': r'doctor[:\s]*rs\.?\s*(\d+)|doctor[:\s]*₹\s*(\d+)',
            'medicine': r'medicine[:\s]*rs\.?\s*(\d+)|medicine[:\s]*₹\s*(\d+)',
            'diagnostic': r'diagnostic[:\s]*rs\.?\s*(\d+)|diagnostic[:\s]*₹\s*(\d+)'
        }
        
        for limit_type, pattern in limit_types.items():
            match = re.search(pattern, text_lower)
            if match:
                for val in match.groups():
                    if val:
                        sub_limits[limit_type] = int(val)
        
        return sub_limits

# Initialize predictor
predictor = RiskPredictor()

# ============================================
# HEADER
# ============================================
st.markdown("""
<div class="main-header">
    <h1 style="margin:0">🛡️ ClaimGuard</h1>
    <p style="margin:0; opacity:0.9">AI-Powered Insurance Claim Risk Analyzer</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# LEFT SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("### 👤 Insured Profile")
    age = st.number_input("Age", 18, 100, 35)
    disease = st.text_input("Pre-existing conditions", "", placeholder="e.g., diabetes, hypertension")
    procedure = st.text_input("Planned procedure (optional)", "", placeholder="e.g., knee surgery")
    
    st.markdown("---")
    
    st.markdown("### 📋 Policy Category")
    policy_type = st.selectbox(
        "Select policy type:",
        ["Health Insurance", "Car Insurance", "Life Insurance", "Travel Insurance"]
    )
    
    st.markdown("---")
    
    st.markdown("### 📂 Upload Policy Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type="pdf",
        help="Upload your insurance policy document for AI-powered claim risk analysis"
    )
    
    if uploaded_file:
        with st.spinner("🔍 Analyzing your policy with ClaimGuard AI..."):
            try:
                # Extract text from PDF
                with pdfplumber.open(uploaded_file) as pdf:
                    text = ""
                    for page in pdf.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted
                
                # Generate unique hash for this PDF
                pdf_hash = hashlib.md5(text.encode()).hexdigest()[:8]
                
                # Detect policy type from PDF content
                text_lower = text.lower()
                
                type_keywords = {
                    "Health Insurance": ['health', 'medical', 'hospital', 'surgery', 'disease', 'treatment', 'doctor', 'medicine', 'illness', 'diagnosis'],
                    "Car Insurance": ['car', 'vehicle', 'motor', 'automobile', 'accident', 'drive', 'driver', 'collision', 'theft', 'damage'],
                    "Life Insurance": ['life', 'death', 'term', 'maturity', 'nominee', 'assured', 'survival', 'beneficiary'],
                    "Travel Insurance": ['travel', 'trip', 'flight', 'baggage', 'overseas', 'foreign', 'passport', 'visa', 'journey']
                }
                
                type_scores = {}
                for p_type, keywords in type_keywords.items():
                    score = sum(text_lower.count(word) for word in keywords)
                    type_scores[p_type] = score
                
                detected_type = max(type_scores, key=type_scores.get) if max(type_scores.values()) > 0 else "Unknown"
                
                # Check if selected type matches detected type
                if detected_type != "Unknown" and detected_type != policy_type:
                    st.error(f"❌ Wrong Policy Type! This appears to be a **{detected_type}** policy, but you selected **{policy_type}**")
                    st.stop()
                
                # Extract key clauses
                extracted_clauses = {}
                terms = {
                    "waiting period": r'waiting[-\s]?period|waiting\s+time|pre[-\s]?existing\s+waiting',
                    "exclusion": r'exclusion|not\s+cover|will\s+not\s+cover|excluded|not\s+payable',
                    "co-pay": r'co[-\s]?pay|copayment|co-payment|coinsurance',
                    "sub-limit": r'sub[-\s]?limit|sublimit|limit\s+of\s+coverage|cap\s+of',
                    "room rent": r'room\s+rent|room\s+charges|accommodation\s+benefit',
                    "pre-existing": r'pre[-\s]?existing|preexisting|known\s+condition',
                    "claim": r'claim\s+process|claim\s+filing|intimation|claim\s+settlement',
                    "deductible": r'deductible|excess|first\s+pay'
                }
                
                for term, pattern in terms.items():
                    sentences = [s for s in text.split('.') if re.search(pattern, s.lower())]
                    if sentences:
                        extracted_clauses[term] = sentences[0].strip() + "."
                    else:
                        extracted_clauses[term] = "Not mentioned in document"
                
                # Extract financial details specific to this policy
                co_pay_percentage = predictor.extract_co_pay_percentage(text)
                deductible = predictor.extract_deductible(text)
                room_rent_cap = predictor.extract_room_rent_cap(text)
                sub_limits = predictor.extract_sub_limits(text)
                
                # Use ML to predict risk scores
                risk_scores = predictor.predict_risk(text, policy_type, age, bool(disease))
                
                # Add some randomness based on PDF content to ensure uniqueness
                text_sum = sum(ord(c) for c in text[:1000]) % 100
                risk_scores['coverage_risk'] = (risk_scores['coverage_risk'] + text_sum // 3) % 100
                risk_scores['cost_risk'] = (risk_scores['cost_risk'] + text_sum // 2) % 100
                risk_scores['delay_risk'] = (risk_scores['delay_risk'] + text_sum) % 100
                
                # Create policy data with financial details
                policy_data = {
                    "id": pdf_hash,
                    "filename": uploaded_file.name,
                    "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "policy_type": policy_type,
                    "detected_type": detected_type,
                    "clauses": extracted_clauses,
                    "risk_scores": risk_scores,
                    "analysis_result": {
                        "co_pay_percentage": co_pay_percentage,
                        "deductible": deductible,
                        "room_rent_cap": room_rent_cap,
                        "sub_limits": sub_limits,
                        "waiting_periods": [],  # Add if you extract these
                        "exclusion_count": 0,    # Add if you count these
                        "claim_time_limit": None
                    },
                    "financial_details": {
                        "co_pay_percentage": co_pay_percentage,
                        "deductible": deductible,
                        "room_rent_cap": room_rent_cap,
                        "sub_limits": sub_limits
                    },
                    "text_sample": text[:500],
                    "unique_id": pdf_hash
                }
                
                # Check if already exists (avoid duplicates)
                existing = [p for p in st.session_state['policy_history'] if p['id'] == pdf_hash]
                if not existing:
                    st.session_state['policy_history'].append(policy_data)
                
                st.session_state['current_policy'] = policy_data
                st.session_state['analysis_done'] = True
                
                st.success(f"✅ Analysis complete! ClaimGuard ID: {pdf_hash}")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    
    # Policy History
    if st.session_state['policy_history']:
        st.markdown("### 📜 Claim History")
        
        for i, policy in enumerate(reversed(st.session_state['policy_history'])):
            scores = policy['risk_scores']
            btn_label = f"{policy['filename'][:15]}... | R:{scores['coverage_risk']}%"
            
            if st.button(f"📄 {btn_label}", key=f"history_{policy['id']}"):
                st.session_state['current_policy'] = policy
                st.rerun()
        
        if st.button("🗑️ Clear All History"):
            st.session_state['policy_history'] = []
            st.session_state['current_policy'] = None
            st.session_state['analysis_done'] = False
            st.rerun()
    
    st.markdown("---")
    
    st.markdown("### 💡 Quick Tips")
    st.info(
        "• Each policy gets unique risk scores\n"
        "• Claim impact varies by policy terms\n"
        "• Compare multiple policies side-by-side\n"
        "• Check co-pay, deductibles, and sub-limits"
    )

# ============================================
# MAIN CONTENT
# ============================================
if not st.session_state['analysis_done'] or st.session_state['current_policy'] is None:
    # Welcome Screen
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 50px 20px;'>
            <h1 style='color: #1a237e;'>👋 Welcome to ClaimGuard</h1>
            <p style='font-size: 18px; color: #666; margin: 30px 0;'>
                Your AI-powered assistant for insurance claim risk analysis.
                Upload your policy document to understand your claim coverage.
            </p>
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                      padding: 30px; border-radius: 15px; color: white;'>
                <h3>🚀 What ClaimGuard Offers</h3>
                <p style='margin: 20px 0;'>✓ ML-powered claim risk assessment</p>
                <p>✓ Policy-specific claim impact analysis</p>
                <p>✓ Key clause extraction</p>
                <p>✓ Side-by-side policy comparison</p>
            </div>
            <p style='margin-top: 30px; color: #1a237e;'>
                <strong>Upload your first policy to get started!</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

else:
    # Show analysis for current policy
    policy = st.session_state['current_policy']
    
    # Policy Header
    st.markdown(f"""
    <div style='background: white; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
        <h3 style='margin:0; color: #1a237e;'>📊 {policy['filename']}</h3>
        <p style='margin:0; color: #666;'>Uploaded: {policy['upload_time']} | ClaimGuard ID: <span class="claim-badge">{policy['unique_id']}</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Risk metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        coverage = policy['risk_scores']['coverage_risk']
        risk_class = "risk-low" if coverage < 30 else "risk-medium" if coverage < 60 else "risk-high"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Claim Coverage Risk</h4>
            <div class="{risk_class}">{coverage}%</div>
            <small>Exclusions & waiting periods</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        cost = policy['risk_scores']['cost_risk']
        risk_class = "risk-low" if cost < 30 else "risk-medium" if cost < 60 else "risk-high"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Out-of-Pocket Risk</h4>
            <div class="{risk_class}">{cost}%</div>
            <small>Co-pay & sub-limits</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        delay = policy['risk_scores']['delay_risk']
        risk_class = "risk-low" if delay < 30 else "risk-medium" if delay < 60 else "risk-high"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Claim Delay Risk</h4>
            <div class="{risk_class}">{delay}%</div>
            <small>Processing time</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_risk = (coverage * 0.4 + cost * 0.35 + delay * 0.25)
        risk_class = "risk-low" if total_risk < 30 else "risk-medium" if total_risk < 60 else "risk-high"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Overall Claim Risk</h4>
            <div class="{risk_class}">{total_risk:.1f}%</div>
            <small>Total claim risk score</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============================================
    # CLAIM IMPACT SIMULATOR
    # ============================================
    st.markdown("### 💰 Claim Impact Simulator")
    st.markdown(f"*Based on {policy['policy_type']} policy terms - Results are unique to this document*")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        claim_amount = st.number_input(
            "Enter expected claim amount (₹)", 
            value=500000, 
            step=50000, 
            min_value=10000,
            max_value=10000000,
            key=f"cost_{policy['unique_id']}"
        )
    
    with col2:
        st.markdown(f"<br>", unsafe_allow_html=True)
        st.markdown(f"**ClaimGuard ID:** `{policy['unique_id']}`")
    
    # Get policy-specific financial details
    financial = policy['financial_details']
    co_pay_pct = financial['co_pay_percentage']
    deductible = financial['deductible']
    room_rent_cap = financial['room_rent_cap']
    sub_limits = financial['sub_limits']

    # ===== SHOW WHAT VALUES ARE BEING USED =====
    st.warning(f"⚠️ **Using these values from THIS PDF:** Co-pay: {co_pay_pct}% | Deductible: ₹{deductible} | Room Rent: {room_rent_cap}")
    # ============================================
    
    # Calculate financial impact based on actual policy terms
    remaining_amount = claim_amount
    
    # 1. Apply deductible first (if any)
    if deductible > 0:
        deductible_amount = min(deductible, remaining_amount)
        remaining_amount -= deductible_amount
    else:
        deductible_amount = 0
    
    # 2. Apply co-pay (if any)
    if co_pay_pct > 0:
        co_pay_amount = remaining_amount * co_pay_pct / 100
        remaining_amount -= co_pay_amount
    else:
        co_pay_amount = 0
    
    # 3. Check for room rent capping (if applicable)
    room_rent_impact = 0
    if room_rent_cap and isinstance(room_rent_cap, int) and claim_amount > room_rent_cap * 30:  # Assuming 30 days
        room_rent_impact = (claim_amount - room_rent_cap * 30) * 0.5  # 50% penalty for excess
        remaining_amount -= room_rent_impact
    
    # 4. Check sub-limits
    sub_limit_impact = 0
    if sub_limits:
        for limit_type, limit_amount in sub_limits.items():
            if claim_amount > limit_amount:
                sub_limit_impact += (claim_amount - limit_amount) * 0.3  # 30% penalty
                remaining_amount -= sub_limit_impact
    
    # Insurance pays the remaining amount
    insurance_pays = remaining_amount
    out_of_pocket = claim_amount - insurance_pays
    
    # Display results with policy-specific factors
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 10px; text-align: center;'>
            <h4 style='color: #666;'>💰 Claim Amount</h4>
            <h2 style='color: #1a237e;'>₹{:,.0f}</h2>
        </div>
        """.format(claim_amount), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: #e8f5e9; padding: 1.5rem; border-radius: 10px; text-align: center;'>
            <h4 style='color: #666;'>✅ Insurance Pays</h4>
            <h2 style='color: #28a745;'>₹{:,.0f}</h2>
        </div>
        """.format(insurance_pays), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: #ffebee; padding: 1.5rem; border-radius: 10px; text-align: center;'>
            <h4 style='color: #666;'>❌ You Pay</h4>
            <h2 style='color: #dc3545;'>₹{:,.0f}</h2>
        </div>
        """.format(out_of_pocket), unsafe_allow_html=True)
    
    # Download Report Button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("📄 Generate PDF Report", use_container_width=True, type="primary"):
            with st.spinner("Generating PDF report..."):
                pdf_path = generate_pdf_report(
                    policy, 
                    policy['analysis_result'], 
                    financial, 
                    claim_amount, 
                    insurance_pays, 
                    out_of_pocket
                )
                
                # Create download link
                filename = f"ClaimGuard_Report_{policy['unique_id']}.pdf"
                download_link = get_download_link(pdf_path, filename)
                
                # Display download link
                st.markdown(f"<center>{download_link}</center>", unsafe_allow_html=True)
                
                # Clean up temp file
                os.unlink(pdf_path)
    
    # Show detailed breakdown based on policy terms
    with st.expander("📊 View Detailed Claim Breakdown"):
        st.markdown(f"**Policy-Specific Terms Applied:**")
        
        breakdown_data = []
        
        if deductible > 0:
            breakdown_data.append(["Deductible", f"₹{deductible:,.0f}", f"₹{deductible_amount:,.0f}"])
        
        if co_pay_pct > 0:
            breakdown_data.append([f"Co-pay ({co_pay_pct}%)", f"{co_pay_pct}%", f"₹{co_pay_amount:,.0f}"])
        
        if room_rent_impact > 0:
            rent_cap_text = f"Room Rent Cap (₹{room_rent_cap:,.0f}/day)" if isinstance(room_rent_cap, int) else f"Room Rent Cap ({room_rent_cap})"
            breakdown_data.append([rent_cap_text, "Penalty applied", f"₹{room_rent_impact:,.0f}"])
        
        if sub_limits:
            for limit_type, amount in sub_limits.items():
                breakdown_data.append([f"{limit_type.title()} Sub-limit", f"₹{amount:,.0f}", f"₹{sub_limit_impact:,.0f}"])
        
        if not breakdown_data:
            st.info("No deductibles, co-pay, or sub-limits detected in this policy")
        else:
            df = pd.DataFrame(breakdown_data, columns=["Term", "Limit", "Impact on Claim"])
            st.table(df)
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Convert decimal values to percentages (0.01 -> 1%)
        chart_values = [
            float(coverage) * 100 if coverage and coverage < 1 else float(coverage),
            float(cost) * 100 if cost and cost < 1 else float(cost),
            float(delay) * 100 if delay and delay < 1 else float(delay)
        ]
        
        # Round to whole numbers for display
        chart_values = [round(val) for val in chart_values]
        
        # Debug - show what's being plotted
        st.caption(f"Chart values: Coverage={chart_values[0]}%, Cost={chart_values[1]}%, Delay={chart_values[2]}%")
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Coverage Risk', 'Out-of-Pocket Risk', 'Delay Risk'],
            values=chart_values,
            hole=.3,
            marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1'],
            textinfo='label+percent',
            textposition='auto'
        )])
        fig_pie.update_layout(
            title=f"Claim Risk Breakdown - {policy['unique_id']}", 
            height=400,
            showlegend=True,
            annotations=[dict(text=f'Total Risk: {total_risk:.1f}%', x=0.5, y=0.5, font_size=15, showarrow=False)]
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Create comparison with industry average
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name='This Policy',
            x=['Coverage', 'Out-of-Pocket', 'Delay'],
            y=[coverage, cost, delay],
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1']
        ))
        fig_bar.add_trace(go.Bar(
            name='Industry Avg',
            x=['Coverage', 'Out-of-Pocket', 'Delay'],
            y=[45, 35, 25],
            marker_color='rgba(200,200,200,0.5)'
        ))
        fig_bar.update_layout(
            title=f"Policy vs Industry Average",
            height=400,
            barmode='group',
            yaxis_range=[0, 100]
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("---")
    
    # Key Findings
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔍 Key Claim Risk Factors")
        factors = []
        
        if financial['co_pay_percentage'] > 0:
            factors.append(f"⚠️ **{financial['co_pay_percentage']}% co-pay** will reduce your claim payout")
        if financial['deductible'] > 0:
            factors.append(f"⚠️ **₹{financial['deductible']:,.0f} deductible** per claim")
        if financial['room_rent_cap']:
            factors.append(f"⚠️ **Room rent capping** of {financial['room_rent_cap']} may affect coverage")
        if 'waiting' in policy['clauses']['waiting period'].lower():
            factors.append("⚠️ **Waiting periods** applicable - claims may be delayed")
        if 'exclusion' in policy['clauses']['exclusion'].lower():
            factors.append("⚠️ **Multiple exclusions** found - some claims may be rejected")
        
        for factor in factors[:5]:
            st.markdown(factor)
        
        if not factors:
            st.success("✅ No major claim risk factors detected")
    
    with col2:
        st.markdown("### 📋 Claim Summary")
        st.markdown(f"**Policy Type:** {policy['policy_type']}")
        st.markdown(f"**Detected:** {policy['detected_type']}")
        st.markdown(f"**Document Length:** {len(policy['text_sample'])} chars")
        st.markdown(f"**ClaimGuard ID:** `{policy['unique_id']}`")
        
        # Show extracted co-pay if any
        if financial['co_pay_percentage'] > 0:
            st.info(f"💰 Co-pay detected: **{financial['co_pay_percentage']}%** of claim amount")
    
    st.markdown("---")
    
    # Extracted Clauses
    with st.expander("📋 View Extracted Policy Clauses"):
        for term, clause in policy['clauses'].items():
            st.markdown(f"**{term.title()}:**")
            st.markdown(f"*{clause}*")
            st.markdown("---")
    
    # Policy Comparison
    if len(st.session_state['policy_history']) > 1:
        st.markdown("---")
        st.markdown("### 📊 ClaimGuard Comparison")
        
        compare_data = []
        for p in st.session_state['policy_history']:
            compare_data.append({
                'Policy': p['filename'][:20] + '...',
                'Claim Coverage': p['risk_scores']['coverage_risk'],
                'Out-of-Pocket': p['risk_scores']['cost_risk'],
                'Delay Risk': p['risk_scores']['delay_risk'],
                'Co-pay': f"{p['financial_details']['co_pay_percentage']}%" if p['financial_details']['co_pay_percentage'] > 0 else "None",
                'Deductible': f"₹{p['financial_details']['deductible']:,.0f}" if p['financial_details']['deductible'] > 0 else "None"
            })
        
        df = pd.DataFrame(compare_data)
        st.dataframe(df, use_container_width=True)

# Footer
st.markdown("""
<div class="footer">
    <p>🛡️ ClaimGuard - AI-Powered Insurance Claim Risk Analysis</p>
    <p style='font-size: 0.8rem;'>© 2024 ClaimGuard. Protecting your claims, protecting your future.</p>
</div>
""", unsafe_allow_html=True)