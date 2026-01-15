"""
Authentication Blueprint: Login and Signup with Session Management
"""
from flask import Blueprint, request, render_template, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import os
import json

auth_bp = Blueprint('auth', __name__)

# Simple in-memory user store (for production, use a proper database)
USERS_FILE = "users.json"

def load_users():
    """Load users from file"""
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_users(users):
    """Save users to file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if not username or not password:
            flash('Please provide both username and password', 'error')
            return render_template('login.html', mode='login')
        
        users = load_users()
        
        if username not in users:
            flash('Invalid username or password', 'error')
            return render_template('login.html', mode='login')
        
        if not check_password_hash(users[username]['password'], password):
            flash('Invalid username or password', 'error')
            return render_template('login.html', mode='login')
        
        # Set session
        session['username'] = username
        session['logged_in'] = True
        flash('Logged in successfully!', 'success')
        return redirect(url_for('main.chat'))
    
    # GET request
    if session.get('logged_in'):
        return redirect(url_for('main.chat'))
    
    return render_template('login.html', mode='login')

@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    """User signup"""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        if not username or not password:
            flash('Please provide both username and password', 'error')
            return render_template('login.html', mode='signup')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('login.html', mode='signup')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long', 'error')
            return render_template('login.html', mode='signup')
        
        users = load_users()
        
        if username in users:
            flash('Username already exists', 'error')
            return render_template('login.html', mode='signup')
        
        # Create new user
        users[username] = {
            'password': generate_password_hash(password),
            'created_at': None  # Could add timestamp if needed
        }
        save_users(users)
        
        # Auto-login after signup
        session['username'] = username
        session['logged_in'] = True
        flash('Account created successfully!', 'success')
        return redirect(url_for('main.chat'))
    
    # GET request
    if session.get('logged_in'):
        return redirect(url_for('main.chat'))
    
    return render_template('login.html', mode='signup')

@auth_bp.route('/logout')
def logout():
    """User logout"""
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('auth.login'))

def login_required(f):
    """Decorator to require login"""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            flash('Please log in to access this page', 'error')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function

