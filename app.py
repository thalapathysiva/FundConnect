from flask import Flask, render_template, request, redirect, url_for, session, flash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
from scipy.spatial.distance import cosine
import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management
app.permanent_session_lifetime = datetime.timedelta(days=30)  # Keep user logged in for 30 days

# In-memory storage for users (replace with a database in production)
users = []

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    """
    Generate a BERT embedding for the input text.
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    # Use the [CLS] token embedding as the sentence embedding
    embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return embedding

def calculate_match_score(business_desc, investor_desc, business_industry, investor_industries):
    """
    Calculate the match score between a business and an investor.
    Combines BERT-based text similarity with industry matching.
    """
    # Ensure descriptions are strings, even if None
    business_desc = business_desc or ''
    investor_desc = investor_desc or ''

    # Get BERT embeddings for descriptions
    business_embedding = get_bert_embedding(business_desc)
    investor_embedding = get_bert_embedding(investor_desc)

    # Flatten the embeddings to 1D arrays
    business_embedding = business_embedding.flatten()
    investor_embedding = investor_embedding.flatten()

    # Calculate cosine similarity (1 - cosine distance)
    text_similarity = 1 - cosine(business_embedding, investor_embedding)

    # Check if the business industry matches the investor's preferred industries
    industry_match = 1 if business_industry.lower() in investor_industries.lower() else 0

    # Combine text similarity and industry match with weights
    match_score = 0.7 * text_similarity + 0.3 * industry_match
    return match_score

@app.route('/')
def home():
    username = session.get('username')  # Get the username from the session
    if username:
        # If user is logged in, redirect to the dashboard
        return redirect(url_for('dashboard'))
    return render_template('dashboard.html', username=None)

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get the user for the dashboard
    username = session.get('username')
    
    # For now, we'll just render the index template
    return render_template('index.html', username=username)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        phone = request.form['phone']
        user_type = request.form['user_type']

        # Check if username already exists
        if any(user['username'] == username for user in users):
            flash('Username already exists! Please choose another.', 'danger')
            return render_template('register.html')

        # Add user to the list
        user_data = {
            'username': username,
            'password': password,
            'phone': phone,
            'user_type': user_type,
            'business_info': {},
            'investor_info': {},
            'join_date': datetime.datetime.now().strftime('%Y-%m-%d')
        }

        # Add business-specific fields
        if user_type == 'business_owner':
            user_data['business_info'] = {
                'business_name': request.form.get('business_name', ''),
                'industry': request.form.get('industry', ''),
                'years_in_operation': request.form.get('years_in_operation', ''),
                'short_description': '',  # Will be filled in business_info.html
                'amount_requested': '',
                'purpose_of_money': '',
                'repayment_term': '',
                'interest_rate': '',
                'annual_revenue': '',
                'net_profit_margin': ''
            }

        # Add investor-specific fields
        elif user_type == 'investor':
            user_data['investor_info'] = {
                'investor_name': request.form.get('investor_name', ''),
                'preferred_industries': request.form.get('preferred_industries', ''),
                'typical_investment_range': '',
                'risk_tolerance': '',
                'preferred_loan_terms': '',
                'total_amount_invested': '',
                'number_of_funded_businesses': ''
            }

        users.append(user_data)
        session['username'] = username  # Log the user in
        session.permanent = True  # Make the session persistent

        flash('Registration successful! Please complete your profile.', 'success')

        # Redirect based on user type
        if user_type == 'business_owner':
            return redirect(url_for('business_info'))
        elif user_type == 'investor':
            return redirect(url_for('investor_info'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Find user in the list
        user = next((user for user in users if user['username'] == username and user['password'] == password), None)
        if user:
            session['username'] = username
            session.permanent = True  # Make the session persistent
            flash('Login successful! Welcome back.', 'success')
            return redirect(url_for('dashboard'))
        
        flash('Invalid username or password. Please try again.', 'danger')
        return render_template('login.html')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('home'))

@app.route('/profiles')
def profiles():
    if 'username' not in session:
        flash('Please log in to view profiles.', 'warning')
        return redirect(url_for('login'))

    # Get the current user
    current_user = next((user for user in users if user['username'] == session['username']), None)
    if not current_user:
        flash('User not found. Please log in again.', 'danger')
        return redirect(url_for('login'))

    # Determine which profiles to show based on user type
    if current_user['user_type'] == 'business_owner':
        profiles_to_show = [user for user in users if user['user_type'] == 'investor' and user['investor_info']]
        profile_type = 'Investor Profiles'
    elif current_user['user_type'] == 'investor':
        profiles_to_show = [user for user in users if user['user_type'] == 'business_owner' and user['business_info']]
        profile_type = 'Business Owner Profiles'
    else:
        profiles_to_show = []
        profile_type = 'Profiles'

    # Calculate match scores for the current user
    if current_user['user_type'] == 'business_owner' and current_user['business_info']:
        business_desc = current_user['business_info'].get('short_description', '')
        business_industry = current_user['business_info'].get('industry', '')
        for profile in profiles_to_show:
            if profile['investor_info']:
                investor_desc = profile['investor_info'].get('investment_criteria', '') + " " + profile['investor_info'].get('preferred_industries', '')
                investor_industries = profile['investor_info'].get('preferred_industries', '')
                profile['match_score'] = calculate_match_score(business_desc, investor_desc, business_industry, investor_industries)
    elif current_user['user_type'] == 'investor' and current_user['investor_info']:
        investor_desc = current_user['investor_info'].get('investment_criteria', '') + " " + current_user['investor_info'].get('preferred_industries', '')
        investor_industries = current_user['investor_info'].get('preferred_industries', '')
        for profile in profiles_to_show:
            if profile['business_info']:
                business_desc = profile['business_info'].get('short_description', '')
                business_industry = profile['business_info'].get('industry', '')
                profile['match_score'] = calculate_match_score(business_desc, investor_desc, business_industry, investor_industries)

    return render_template('profiles.html', profiles=profiles_to_show, profile_type=profile_type)

@app.route('/business_info', methods=['GET', 'POST'])
def business_info():
    if 'username' not in session:
        flash('Please log in to complete your profile.', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        # Get values from the form
        short_description = request.form.get('short_description', '')
        amount_requested = request.form.get('amount_requested', '')
        purpose_of_money = request.form.get('purpose_of_money', '')
        repayment_term = request.form.get('repayment_term', '')
        interest_rate = request.form.get('interest_rate', '')
        annual_revenue = request.form.get('annual_revenue', '')
        net_profit_margin = request.form.get('net_profit_margin', '')

        # Add validation here if needed
        if not short_description or not amount_requested or not purpose_of_money:
            flash('Please fill out all required fields.', 'danger')
            return render_template('business_info.html')

        # Update user's business info
        business_info = {
            'short_description': short_description,
            'amount_requested': amount_requested,
            'purpose_of_money': purpose_of_money,
            'repayment_term': repayment_term,
            'interest_rate': interest_rate,
            'annual_revenue': annual_revenue,
            'net_profit_margin': net_profit_margin
        }

        # Find the user and update their business info
        user = next((user for user in users if user['username'] == session['username']), None)
        if user:
            if user['business_info']:
                user['business_info'].update(business_info)
            else:
                user['business_info'] = business_info
                
            flash('Profile information updated successfully!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('User not found. Please log in again.', 'danger')
            return redirect(url_for('login'))

    return render_template('business_info.html')

@app.route('/investor_info', methods=['GET', 'POST'])
def investor_info():
    if 'username' not in session:
        flash('Please log in to complete your profile.', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        # Get values from the form
        typical_investment_range = request.form.get('typical_investment_range', '')
        risk_tolerance = request.form.get('risk_tolerance', '')
        investment_term = request.form.get('investment_term', '')
        preferred_industries = request.form.get('preferred_industries', '')
        investment_criteria = request.form.get('investment_criteria', '')
        min_return_expected = request.form.get('min_return_expected', '')
        max_businesses = request.form.get('max_businesses', '')
        previous_experience = request.form.get('previous_experience', '')

        # Add validation here if needed
        if not typical_investment_range or not risk_tolerance:
            flash('Please fill out all required fields.', 'danger')
            return render_template('investor_info.html')

        # Update user's investor info
        investor_info = {
            'typical_investment_range': typical_investment_range,
            'risk_tolerance': risk_tolerance,
            'investment_term': investment_term,
            'preferred_industries': preferred_industries,
            'investment_criteria': investment_criteria,
            'min_return_expected': min_return_expected,
            'max_businesses': max_businesses,
            'previous_experience': previous_experience
        }

        # Find the user and update their investor info
        user = next((user for user in users if user['username'] == session['username']), None)
        if user:
            if user['investor_info']:
                user['investor_info'].update(investor_info)
            else:
                user['investor_info'] = investor_info
                
            flash('Profile information updated successfully!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('User not found. Please log in again.', 'danger')
            return redirect(url_for('login'))

    return render_template('investor_info.html')

@app.route('/profile/edit', methods=['GET'])
def edit_profile():
    if 'username' not in session:
        flash('Please log in to edit your profile.', 'warning')
        return redirect(url_for('login'))
    
    # Get the current user
    current_user = next((user for user in users if user['username'] == session['username']), None)
    if not current_user:
        flash('User not found. Please log in again.', 'danger')
        return redirect(url_for('login'))
    
    # Redirect to the appropriate profile edit page based on user type
    if current_user['user_type'] == 'business_owner':
        return redirect(url_for('business_info'))
    elif current_user['user_type'] == 'investor':
        return redirect(url_for('investor_info'))
    else:
        flash('Invalid user type. Please contact support.', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/profile/<username>')
def view_profile(username):
    if 'username' not in session:
        flash('Please log in to view profiles.', 'warning')
        return redirect(url_for('login'))
    
    # Get the requested user profile
    profile_user = next((user for user in users if user['username'] == username), None)
    if not profile_user:
        flash('Profile not found.', 'danger')
        return redirect(url_for('profiles'))
    
    # Get the current user
    current_user = next((user for user in users if user['username'] == session['username']), None)
    
    # Calculate match score if viewing compatible profile type
    match_score = None
    if current_user['user_type'] == 'business_owner' and profile_user['user_type'] == 'investor':
        if current_user['business_info'] and profile_user['investor_info']:
            business_desc = current_user['business_info'].get('short_description', '')
            business_industry = current_user['business_info'].get('industry', '')
            investor_desc = profile_user['investor_info'].get('investment_criteria', '') + " " + profile_user['investor_info'].get('preferred_industries', '')
            investor_industries = profile_user['investor_info'].get('preferred_industries', '')
            match_score = calculate_match_score(business_desc, investor_desc, business_industry, investor_industries)
    elif current_user['user_type'] == 'investor' and profile_user['user_type'] == 'business_owner':
        if current_user['investor_info'] and profile_user['business_info']:
            investor_desc = current_user['investor_info'].get('investment_criteria', '') + " " + current_user['investor_info'].get('preferred_industries', '')
            investor_industries = current_user['investor_info'].get('preferred_industries', '')
            business_desc = profile_user['business_info'].get('short_description', '')
            business_industry = profile_user['business_info'].get('industry', '')
            match_score = calculate_match_score(business_desc, investor_desc, business_industry, investor_industries)
    
    return render_template('profile_detail.html', profile=profile_user, match_score=match_score)

@app.route('/contact/<username>', methods=['GET', 'POST'])
def contact_user(username):
    if 'username' not in session:
        flash('Please log in to contact users.', 'warning')
        return redirect(url_for('login'))
    
    # Get the target user
    target_user = next((user for user in users if user['username'] == username), None)
    if not target_user:
        flash('User not found.', 'danger')
        return redirect(url_for('profiles'))
    
    if request.method == 'POST':
        message = request.form.get('message')
        sender = session['username']
        
        # In a real app, you would store this message in a database
        # and/or send an email notification
        flash(f'Your message has been sent to {username}!', 'success')
        return redirect(url_for('profiles'))
    
    return render_template('contact_form.html', recipient=target_user)

@app.route('/matches')
def view_matches():
    if 'username' not in session:
        flash('Please log in to view your matches.', 'warning')
        return redirect(url_for('login'))
    
    # Get the current user
    current_user = next((user for user in users if user['username'] == session['username']), None)
    if not current_user:
        flash('User not found. Please log in again.', 'danger')
        return redirect(url_for('login'))
    
    # Find potential matches based on user type
    matches = []
    threshold = 0.6  # Match score threshold (60%)
    
    if current_user['user_type'] == 'business_owner' and current_user['business_info']:
        business_desc = current_user['business_info'].get('short_description', '')
        business_industry = current_user['business_info'].get('industry', '')
        
        for user in users:
            if user['user_type'] == 'investor' and user['investor_info']:
                investor_desc = user['investor_info'].get('investment_criteria', '') + " " + user['investor_info'].get('preferred_industries', '')
                investor_industries = user['investor_info'].get('preferred_industries', '')
                match_score = calculate_match_score(business_desc, investor_desc, business_industry, investor_industries)
                
                if match_score >= threshold:
                    user_copy = user.copy()
                    user_copy['match_score'] = match_score
                    matches.append(user_copy)
    
    elif current_user['user_type'] == 'investor' and current_user['investor_info']:
        investor_desc = current_user['investor_info'].get('investment_criteria', '') + " " + current_user['investor_info'].get('preferred_industries', '')
        investor_industries = current_user['investor_info'].get('preferred_industries', '')
        
        for user in users:
            if user['user_type'] == 'business_owner' and user['business_info']:
                business_desc = user['business_info'].get('short_description', '')
                business_industry = user['business_info'].get('industry', '')
                match_score = calculate_match_score(business_desc, investor_desc, business_industry, investor_industries)
                
                if match_score >= threshold:
                    user_copy = user.copy()
                    user_copy['match_score'] = match_score
                    matches.append(user_copy)
    
    # Sort matches by match score (descending)
    matches.sort(key=lambda x: x.get('match_score', 0), reverse=True)
    
    return render_template('matches.html', matches=matches)

if __name__ == '__main__':
    app.run(debug=True)


