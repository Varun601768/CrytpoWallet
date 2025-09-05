from flask import Flask, render_template, request, jsonify,render_template_string, redirect, url_for, flash
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import os
import logging
import re
import warnings
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///crypto_users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship to prediction history
    predictions = db.relationship('PredictionHistory', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

# Prediction History Model
class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    ticker = db.Column(db.String(20), nullable=False)
    currency_code = db.Column(db.String(10), nullable=False)
    prediction_days = db.Column(db.Integer, nullable=False)
    latest_price = db.Column(db.Float, nullable=False)
    future_predictions = db.Column(db.Text, nullable=False)  # JSON string of predictions
    future_dates = db.Column(db.Text, nullable=False)  # JSON string of dates
    historical_dates = db.Column(db.Text, nullable=False)  # JSON string of historical dates
    historical_prices = db.Column(db.Text, nullable=False)  # JSON string of historical prices
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<PredictionHistory {self.ticker} by {self.user_id}>'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Email configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_ADDRESS = "varunmcchinthu@gmail.com"  # Replace with your Gmail
EMAIL_PASSWORD = "uacrivcemeqtrtez"    # Replace with your Gmail App Password
RECIPIENT_EMAIL = "varunmcchinthu@gmail.com"

# Mock exchange rates
EXCHANGE_RATES = {
    'USD': 1.0,
    'INR': 83.5,
    'EUR': 0.93,
    'JPY': 149.2,
    'GBP': 0.79,
    'CAD': 1.35
}

# Model file path
MODEL_PATH = 'crypto_model.h5'

# Popular crypto tickers for validation
POPULAR_TICKERS = [
    'BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD', 'DOT-USD',
    'MATIC-USD', 'LTC-USD', 'LINK-USD', 'XRP-USD', 'DOGE-USD',
    'AVAX-USD', 'ATOM-USD', 'ALGO-USD', 'ICP-USD', 'UNI-USD'
]

def validate_ticker(ticker):
    """Improved ticker validation with better error handling"""
    if not ticker:
        return False, "Ticker cannot be empty"
    
    ticker = ticker.upper()
    
    # Check format
    pattern = r'^[A-Z0-9]{2,10}-USD$'
    if not re.match(pattern, ticker):
        return False, f"Invalid ticker format: {ticker}. Must be like BTC-USD or ETH-USD"
    
    try:
        # Quick test fetch to validate ticker exists
        test_data = yf.download(ticker, period='5d', progress=False)
        if test_data.empty:
            return False, f"No data available for ticker: {ticker}"
        return True, "Valid ticker"
    except Exception as e:
        logger.error(f"Error validating ticker {ticker}: {e}")
        return False, f"Error validating ticker: {str(e)}"

def fetch_crypto_data_for_analysis(ticker, start_date, end_date):
    """Fetch cryptocurrency data for analysis with improved date handling"""
    try:
        logger.info(f"Fetching analysis data for {ticker} from {start_date} to {end_date}")
        
        # Ensure dates are datetime objects
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Limit end date to current date if it's in the future
        current_date = datetime.now()
        if end_date > current_date:
            end_date = current_date
        
        # Fetch data from Yahoo Finance
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            logger.warning(f"No data available for {ticker} in the specified date range")
            return None
        
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns.values]
        
        # Ensure we have required columns
        required_cols = ['Close', 'Volume', 'High', 'Low', 'Open']
        available_cols = [col for col in required_cols if col in data.columns]
        
        if 'Close' not in available_cols:
            logger.error(f"Missing required 'Close' column for {ticker}")
            return None
        
        # Clean data and handle missing values
        data = data[available_cols].dropna()
        
        logger.info(f"Successfully fetched {len(data)} rows of data for {ticker}")
        return data
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None

def generate_historical_analysis_data(ticker, start_year=2015, end_year=2025):
    """Generate historical and predicted data for analysis with improved date handling"""
    try:
        # Validate input years
        current_year = datetime.now().year
        if start_year > current_year:
            return None
        
        # Set date range
        start_date = datetime(start_year, 1, 1)
        
        # For end year, use current date if end_year is current year or future
        if end_year >= current_year:
            end_date = datetime.now()
        else:
            end_date = datetime(end_year, 12, 31)
        
        # Fetch real historical data
        historical_data = fetch_crypto_data_for_analysis(ticker, start_date, end_date)
        
        if historical_data is None or historical_data.empty:
            logger.error(f"No historical data available for {ticker}")
            return None
        
        # Resample to monthly data for better visualization (optional)
        # For more granular data, you can use daily data
        if len(historical_data) > 1000:  # If too many data points, resample monthly
            monthly_data = historical_data.resample('M').last()
        else:
            monthly_data = historical_data
        
        result_data = []
        
        # Add historical data
        for date, row in monthly_data.iterrows():
            result_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'price': float(row['Close']),
                'volume': float(row.get('Volume', 0)),
                'high': float(row.get('High', row['Close'])),
                'low': float(row.get('Low', row['Close'])),
                'open': float(row.get('Open', row['Close'])),
                'type': 'historical'
            })
        
        # Generate future predictions if end_year is beyond current year
        if end_year > current_year:
            future_data = generate_future_predictions(monthly_data, end_year)
            result_data.extend(future_data)
        
        logger.info(f"Generated {len(result_data)} data points for {ticker}")
        return result_data
        
    except Exception as e:
        logger.error(f"Error generating analysis data for {ticker}: {e}")
        return None

def generate_future_predictions(historical_data, end_year):
    """Generate future predictions based on historical trends"""
    try:
        if historical_data.empty:
            return []
        
        # Get the last known price and calculate trend
        last_price = historical_data['Close'].iloc[-1]
        prices = historical_data['Close'].values
        
        # Calculate simple trend (you can make this more sophisticated)
        if len(prices) > 12:  # Need at least 12 data points
            recent_prices = prices[-12:]  # Last 12 periods
            trend = (recent_prices[-1] - recent_prices[0]) / len(recent_prices)
        else:
            trend = 0
        
        # Generate future predictions
        future_data = []
        current_date = historical_data.index[-1]
        current_price = last_price
        current_year = datetime.now().year
        
        # Generate monthly predictions until end_year
        months_to_predict = (end_year - current_year) * 12
        
        for i in range(1, months_to_predict + 1):
            # Add trend with some randomness
            volatility = np.std(prices[-12:]) if len(prices) > 12 else last_price * 0.1
            random_factor = np.random.normal(0, volatility * 0.1)
            
            predicted_price = current_price + trend + random_factor
            
            # Ensure price doesn't go negative
            predicted_price = max(predicted_price, last_price * 0.1)
            
            # Generate future date
            future_date = current_date + timedelta(days=30 * i)
            
            future_data.append({
                'date': future_date.strftime('%Y-%m-%d'),
                'price': float(predicted_price),
                'volume': float(historical_data['Volume'].iloc[-1]) if 'Volume' in historical_data.columns else 0,
                'high': float(predicted_price * 1.05),
                'low': float(predicted_price * 0.95),
                'open': float(predicted_price * 0.98),
                'type': 'predicted'
            })
            
            current_price = predicted_price
        
        return future_data
        
    except Exception as e:
        logger.error(f"Error generating future predictions: {e}")
        return []

def fetch_crypto_data(ticker, start_date, end_date):
    """Original fetch function for prediction model"""
    try:
        logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
        
        # Try different time periods if initial fetch fails
        periods_to_try = [
            (start_date, end_date),
            (datetime.now() - timedelta(days=365), end_date),
            (datetime.now() - timedelta(days=180), end_date),
            (datetime.now() - timedelta(days=90), end_date)
        ]
        
        data = None
        for start, end in periods_to_try:
            try:
                data = yf.download(ticker, start=start, end=end, progress=False)
                if not data.empty:
                    logger.info(f"Successfully fetched {len(data)} rows for period {start} to {end}")
                    break
            except Exception as e:
                logger.warning(f"Failed to fetch data for period {start} to {end}: {e}")
                continue
        
        if data is None or data.empty:
            logger.error(f"No data available for {ticker} in any time period")
            return None
        
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns.values]
        
        # Ensure we have required columns
        required_cols = ['Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return None
        
        # Clean data
        data = data[required_cols].dropna()
        
        if len(data) < 30:
            logger.error(f"Insufficient data: only {len(data)} rows available")
            return None
            
        logger.info(f"Successfully processed {len(data)} rows of data")
        return data
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None

def get_exchange_rate(from_currency, to_currency):
    """Get exchange rate between currencies"""
    try:
        if from_currency == to_currency:
            return 1.0
        return EXCHANGE_RATES.get(to_currency, 1.0) / EXCHANGE_RATES.get(from_currency, 1.0)
    except Exception as e:
        logger.warning(f"Error getting exchange rate from {from_currency} to {to_currency}: {e}")
        return 1.0

def create_model(input_shape):
    """Create LSTM model for crypto price prediction"""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def get_or_create_model(input_shape):
    """Load existing model or create new one"""
    if os.path.exists(MODEL_PATH):
        try:
            logger.info(f"Loading existing model from {MODEL_PATH}")
            return load_model(MODEL_PATH)
        except Exception as e:
            logger.warning(f"Failed to load model: {e}. Creating new model.")
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
    
    logger.info("Creating new model")
    return create_model(input_shape)

def add_technical_indicators(data):
    """Add technical indicators to improve prediction accuracy"""
    # Simple Moving Averages
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    
    # Price changes
    data['Price_Change'] = data['Close'].pct_change()
    data['Volume_Change'] = data['Volume'].pct_change()
    
    # Fill NaN values
    data = data.fillna(method='bfill').fillna(0)
    
    return data

def predict_crypto_prices(ticker, prediction_days, currency_code):
    """Main prediction function with improved error handling"""
    try:
        # Validate inputs
        if prediction_days < 1 or prediction_days > 30:
            return None, None, None, None, None, "Prediction days must be between 1 and 30"
        
        # Validate ticker
        is_valid, message = validate_ticker(ticker)
        if not is_valid:
            return None, None, None, None, None, message
        
        # Set date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Fetch data
        stock_data = fetch_crypto_data(ticker, start_date, end_date)
        if stock_data is None:
            return None, None, None, None, None, f"Failed to fetch data for {ticker}. Please try a different ticker."
        
        # Add technical indicators
        stock_data = add_technical_indicators(stock_data)
        
        # Currency conversion
        exchange_rate = get_exchange_rate('USD', currency_code)
        stock_data['Close'] = stock_data['Close'] * exchange_rate
        
        # Get latest price
        latest_price = stock_data['Close'].iloc[-1]
        
        # Prepare features
        feature_cols = ['Close', 'Volume', 'SMA_5', 'SMA_20', 'Price_Change', 'Volume_Change']
        features_data = stock_data[feature_cols].values
        
        # Check data sufficiency
        sequence_length = 60
        if len(features_data) < sequence_length + 10:
            return None, None, None, None, None, f"Insufficient data for {ticker}. Need at least {sequence_length + 10} days of data."
        
        # Scale features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(features_data)
        
        # Create sequences for training
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, 0])
        
        if len(X) == 0:
            return None, None, None, None, None, "Insufficient data to create training sequences"
        
        X, y = np.array(X), np.array(y)
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Get or create model
        model = get_or_create_model((X_train.shape[1], X_train.shape[2]))
        
        # Train model if it's new
        if not os.path.exists(MODEL_PATH):
            logger.info("Training model...")
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            history = model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=50,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],
                verbose=0
            )
            
            model.save(MODEL_PATH)
            logger.info(f"Model trained and saved to {MODEL_PATH}")
        
        # Make predictions
        last_sequence = scaled_data[-sequence_length:]
        future_predictions = []
        
        current_sequence = last_sequence.copy()
        
        for day in range(prediction_days):
            # Reshape for prediction
            input_seq = current_sequence.reshape(1, sequence_length, len(feature_cols))
            
            # Predict next day
            next_pred = model.predict(input_seq, verbose=0)[0, 0]
            future_predictions.append(next_pred)
            
            # Create next input sequence
            next_row = current_sequence[-1].copy()
            next_row[0] = next_pred
            
            # Shift sequence and add new prediction
            current_sequence = np.vstack([current_sequence[1:], next_row])
        
        # Inverse transform predictions
        future_predictions = np.array(future_predictions)
        
        # Create dummy array for inverse transform
        dummy_preds = np.zeros((len(future_predictions), len(feature_cols)))
        dummy_preds[:, 0] = future_predictions
        
        # Inverse transform
        future_prices = scaler.inverse_transform(dummy_preds)[:, 0]
        
        # Prepare historical data for charts
        chart_days = min(60, len(stock_data))
        historical_dates = stock_data.index[-chart_days:].strftime('%Y-%m-%d').tolist()
        historical_prices = stock_data['Close'][-chart_days:].tolist()
        
        # Future dates
        future_dates = []
        current_date = stock_data.index[-1]
        for i in range(1, prediction_days + 1):
            future_date = current_date + timedelta(days=i)
            future_dates.append(future_date.strftime('%Y-%m-%d'))
        
        logger.info(f"Successfully generated predictions for {ticker}")
        return latest_price, future_prices, historical_dates, historical_prices, future_dates, None
        
    except Exception as e:
        logger.error(f"Error in predict_crypto_prices: {e}")
        return None, None, None, None, None, f"Prediction error: {str(e)}"

def save_prediction_history(user_id, ticker, currency_code, prediction_days, latest_price, 
                          future_predictions, future_dates, historical_dates, historical_prices):
    """Save prediction history to database"""
    try:
        # Convert lists to JSON strings
        future_predictions_json = json.dumps([float(p) for p in future_predictions])
        future_dates_json = json.dumps(future_dates)
        historical_dates_json = json.dumps(historical_dates)
        historical_prices_json = json.dumps([float(p) for p in historical_prices])
        
        # Create new prediction history record
        prediction_record = PredictionHistory(
            user_id=user_id,
            ticker=ticker,
            currency_code=currency_code,
            prediction_days=prediction_days,
            latest_price=float(latest_price),
            future_predictions=future_predictions_json,
            future_dates=future_dates_json,
            historical_dates=historical_dates_json,
            historical_prices=historical_prices_json
        )
        
        # Save to database
        db.session.add(prediction_record)
        db.session.commit()
        
        logger.info(f"Saved prediction history for user {user_id}, ticker {ticker}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving prediction history: {e}")
        db.session.rollback()
        return False

# Authentication Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Please fill in all fields.', 'error')
            return render_template('login.html')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user, remember=True)
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            else:
                flash(f'Welcome back, {user.first_name}!', 'success')
                return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        
        # Validation
        if not all([username, email, password, confirm_password, first_name, last_name]):
            flash('Please fill in all fields.', 'error')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'error')
            return render_template('register.html')
        
        # Check if user already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists.', 'error')
            return render_template('register.html')
        
        existing_email = User.query.filter_by(email=email).first()
        if existing_email:
            flash('Email already registered.', 'error')
            return render_template('register.html')
        
        # Create new user
        new_user = User(
            username=username,
            email=email,
            first_name=first_name,
            last_name=last_name
        )
        new_user.set_password(password)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('Registration failed. Please try again.', 'error')
            logger.error(f"Registration error: {e}")
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# Routes
@app.route('/')
@login_required
def home():
    return render_template('home.html')

@app.route('/analysis')
@login_required
def analysis():
    return render_template('analysis.html',result="Test value")

@app.route('/con')
@login_required
def contact():
    return render_template('contact.html')

@app.route('/ab')
@login_required
def about():
    return render_template('about.html')

@app.route('/profile')
@login_required
def profile():
    """Display user profile with prediction history"""
    try:
        logger.info(f"Profile route accessed by user: {current_user.username}")
        
        # Get user's prediction history, ordered by most recent first
        predictions = PredictionHistory.query.filter_by(user_id=current_user.id)\
                                            .order_by(PredictionHistory.created_at.desc())\
                                            .all()
        
        logger.info(f"Found {len(predictions)} predictions for user {current_user.id}")
        
        # Process prediction data for display
        processed_predictions = []
        for pred in predictions:
            try:
                # Parse JSON data
                future_predictions = json.loads(pred.future_predictions)
                future_dates = json.loads(pred.future_dates)
                historical_dates = json.loads(pred.historical_dates)
                historical_prices = json.loads(pred.historical_prices)
                
                # Create predictions list for display
                predictions_list = list(zip(future_dates, future_predictions))
                
                processed_predictions.append({
                    'id': pred.id,
                    'ticker': pred.ticker,
                    'currency_code': pred.currency_code,
                    'prediction_days': pred.prediction_days,
                    'latest_price': pred.latest_price,
                    'predictions': predictions_list,
                    'historical_dates': historical_dates,
                    'historical_prices': historical_prices,
                    'future_dates': future_dates,
                    'future_predictions': future_predictions,
                    'created_at': pred.created_at.strftime('%Y-%m-%d %H:%M:%S')
                })
            except Exception as e:
                logger.error(f"Error processing prediction {pred.id}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(processed_predictions)} predictions")
        
        return render_template('profile.html', 
                             user=current_user, 
                             predictions=processed_predictions)
                             
    except Exception as e:
        logger.error(f"Error in profile route: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        flash('Error loading profile data.', 'error')
        return redirect(url_for('home'))

@app.route('/delete-prediction/<int:prediction_id>', methods=['POST'])
@login_required
def delete_prediction(prediction_id):
    """Delete a specific prediction from user's history"""
    try:
        # Find the prediction and ensure it belongs to the current user
        prediction = PredictionHistory.query.filter_by(
            id=prediction_id, 
            user_id=current_user.id
        ).first()
        
        if not prediction:
            flash('Prediction not found or you do not have permission to delete it.', 'error')
            return redirect(url_for('profile'))
        
        # Delete the prediction
        db.session.delete(prediction)
        db.session.commit()
        
        flash('Prediction deleted successfully.', 'success')
        return redirect(url_for('profile'))
        
    except Exception as e:
        logger.error(f"Error deleting prediction {prediction_id}: {e}")
        flash('Error deleting prediction.', 'error')
        return redirect(url_for('profile'))

@app.route('/send-email', methods=['POST'])
def send_email():
    try:
        # Get form data
        data = request.get_json()
        
        first_name = data.get('firstName')
        last_name = data.get('lastName')
        email = data.get('email')
        phone = data.get('phone', 'Not provided')
        subject = data.get('subject')
        message = data.get('message')
        
        # Create email content
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = f"New Contact Form Submission: {subject}"
        
        # Email body
        body = f"""
        New Contact Form Submission
        
        Name: {first_name} {last_name}
        Email: {email}
        Phone: {phone}
        Subject: {subject}
        
        Message:
        {message}
        
        Submitted on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        text = msg.as_string()
        server.sendmail(EMAIL_ADDRESS, RECIPIENT_EMAIL, text)
        server.quit()
        
        return jsonify({'status': 'success', 'message': 'Email sent successfully'})
        
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Failed to send email'}), 500


@app.route('/api/analysis/<ticker>')
def get_analysis_data(ticker):
    """API endpoint for getting historical analysis data with improved error handling"""
    try:
        # Get parameters with validation
        start_year = request.args.get('start_year', 2015, type=int)
        end_year = request.args.get('end_year', 2025, type=int)
        
        # Validate years
        current_year = datetime.now().year
        if start_year < 2010 or start_year > current_year:
            return jsonify({'error': f'Start year must be between 2010 and {current_year}'}), 400
        
        if end_year < start_year:
            return jsonify({'error': 'End year must be greater than or equal to start year'}), 400
        
        if end_year > current_year + 10:
            return jsonify({'error': f'End year cannot be more than {current_year + 10}'}), 400
        
        # Validate ticker
        is_valid, message = validate_ticker(ticker)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Generate analysis data
        data = generate_historical_analysis_data(ticker, start_year, end_year)
        
        if data is None or len(data) == 0:
            return jsonify({'error': f'No data available for {ticker} in the specified date range'}), 404
        
        # Return successful response
        return jsonify({
            'data': data,
            'ticker': ticker,
            'start_year': start_year,
            'end_year': end_year,
            'total_points': len(data)
        })
        
    except Exception as e:
        logger.error(f"Error in analysis API: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        ticker = request.form.get('ticker', 'ETH-USD').upper()
        
        try:
            prediction_days = int(request.form.get('prediction_days', 7))
            if prediction_days < 1 or prediction_days > 30:
                raise ValueError("Prediction days must be between 1 and 30")
        except ValueError as e:
            return render_template('index.html', 
                                 error=str(e), 
                                 currency_code='USD', 
                                 ticker=ticker, 
                                 prediction_days=7,
                                 popular_tickers=POPULAR_TICKERS)
        
        currency_code = request.form.get('currency', 'USD')
        
        # Get predictions
        latest_price, future_predictions, historical_dates, historical_prices, future_dates, error = \
            predict_crypto_prices(ticker, prediction_days, currency_code)
        
        if error:
            return render_template('index.html', 
                                 error=error,
                                 currency_code=currency_code, 
                                 ticker=ticker, 
                                 prediction_days=prediction_days,
                                 popular_tickers=POPULAR_TICKERS)
        
        # Prepare predictions for template
        predictions = list(zip(future_dates, future_predictions))
        
        # Save prediction history to database
        save_prediction_history(
            current_user.id, ticker, currency_code, prediction_days, latest_price,
            future_predictions, future_dates, historical_dates, historical_prices
        )
        
        return render_template('index.html',
                             ticker=ticker,
                             latest_price=round(latest_price, 2),
                             predictions=[(date, round(price, 2)) for date, price in predictions],
                             currency_code=currency_code,
                             prediction_days=prediction_days,
                             historical_dates=historical_dates,
                             historical_prices=[round(p, 2) for p in historical_prices],
                             future_dates=future_dates,
                             future_predictions=[round(p, 2) for p in future_predictions],
                             popular_tickers=POPULAR_TICKERS,
                             success=True)
    
    return render_template('index.html', 
                         currency_code='USD', 
                         ticker='ETH-USD', 
                         prediction_days=7,
                         popular_tickers=POPULAR_TICKERS)

@app.route('/result', methods=['POST'])
@login_required
def result():
    try:
        ticker = request.form['ticker']
        currency_code = request.form['currency_code']
        historical_dates = request.form['historical_dates'].split(',')
        historical_prices = [float(p) for p in request.form['historical_prices'].split(',')]
        future_dates = request.form['future_dates'].split(',')
        future_predictions = [float(p) for p in request.form['future_predictions'].split(',')]
        predictions = list(zip(future_dates, future_predictions))
        
        return render_template('result.html',
                             ticker=ticker,
                             currency_code=currency_code,
                             historical_dates=historical_dates,
                             historical_prices=historical_prices,
                             future_dates=future_dates,
                             future_predictions=future_predictions,
                             predictions=predictions)
                             
    except Exception as e:
        logger.error(f"Error in result route: {e}")
        return render_template('index.html', 
                             error=f"Error processing results: {e}",
                             currency_code='USD', 
                             ticker='ETH-USD', 
                             prediction_days=7,
                             popular_tickers=POPULAR_TICKERS)

@app.route('/result/<int:prediction_id>', methods=['GET'])
@login_required
def result_by_id(prediction_id):
    try:
        prediction = PredictionHistory.query.filter_by(
            id=prediction_id,
            user_id=current_user.id
        ).first()

        if not prediction:
            flash('Prediction not found or access denied.', 'error')
            return redirect(url_for('profile'))

        future_predictions = json.loads(prediction.future_predictions)
        future_dates = json.loads(prediction.future_dates)
        historical_dates = json.loads(prediction.historical_dates)
        historical_prices = json.loads(prediction.historical_prices)
        predictions = list(zip(future_dates, future_predictions))

        return render_template(
            'result.html',
            ticker=prediction.ticker,
            currency_code=prediction.currency_code,
            historical_dates=historical_dates,
            historical_prices=historical_prices,
            future_dates=future_dates,
            future_predictions=future_predictions,
            predictions=predictions
        )
    except Exception as e:
        logger.error(f"Error loading prediction {prediction_id}: {e}")
        flash('Error loading prediction details.', 'error')
        return redirect(url_for('profile'))

if __name__ == '__main__':
    # Create tables if they don't exist
    with app.app_context():
        db.create_all()
        logger.info("Database tables checked/created")
    app.run(debug=True, host='0.0.0.0', port=5000)
