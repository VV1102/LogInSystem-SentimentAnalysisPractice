from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.utils import secure_filename
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///SentimentDataBase.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = '9bec198ba34ffb8e1e812c0234c71783'
db = SQLAlchemy(app)
migrate = Migrate(app, db)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    past_entries = db.relationship('AnalysisEntry', backref='user', lazy=True)

class AnalysisEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    product_id = db.Column(db.String(50), nullable=False)
    timeframe = db.Column(db.String(50), nullable=False)
    plot_filename = db.Column(db.String(100), nullable=False)
    positive_count = db.Column(db.Integer, nullable=False)
    neutral_count = db.Column(db.Integer, nullable=False)
    negative_count = db.Column(db.Integer, nullable=False)

def train_model():
    df = pd.read_csv('Reviews.csv')
    df = df[df['Score'].notnull()]
    X = df['Text']
    y = df['Score'].apply(lambda x: 1 if x > 3 else (-1 if x < 3 else 0))
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    joblib.dump((model, vectorizer), 'reviews_analysis_model.pkl')

def analyze_reviews(csv_file_path, product_id, start_time, end_time):
    model, vectorizer = joblib.load('reviews_analysis_model.pkl')
    df = pd.read_csv(csv_file_path)
    start_time = int(start_time)
    end_time = int(end_time)
    df['Time'] = pd.to_datetime(df['Time'], unit='s')  # Convert 'Time' column to datetime if necessary
    filtered_data = df[(df['ProductId'] == product_id) & (df['Time'] >= start_time) & (df['Time'] <= end_time)]
    if filtered_data.empty:
        return None, 0, 0, 0
    X = vectorizer.transform(filtered_data['Text'])
    predictions = model.predict(X)
    positive_count = (predictions == 1).sum()
    neutral_count = (predictions == 0).sum()
    negative_count = (predictions == -1).sum()
    plt.figure(figsize=(10, 4))
    sns.barplot(x=['Positive', 'Neutral', 'Negative'], y=[positive_count, neutral_count, negative_count])
    plt.title(f'Product ID: {product_id} - Time frame: {start_time} to {end_time}')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    if not os.path.exists('static/plots'):
        os.makedirs('static/plots')
    plot_filename = f'static/plots/count_plot_{product_id}_{start_time}_{end_time}.png'
    plt.savefig(plot_filename)
    plt.close()
    return plot_filename, positive_count, neutral_count, negative_count

@app.route("/", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session['user_id'] = user.id
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash('Account with this email already exists')
            return redirect(url_for('register'))
        new_user = User(username=username, password=password, email=email)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful. Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.')
    return redirect(url_for('login'))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        if 'csv_file' not in request.files:
            flash('No file part')
            return redirect(url_for('dashboard'))
        file = request.files['csv_file']
        if file.filename == '':
            flash('No selected file')
            return redirect(url_for('dashboard'))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            product_id = request.form['product_id']
            start_time = request.form['start_time']
            end_time = request.form['end_time']
            if not product_id or not start_time or not end_time:
                flash('Missing required fields')
                return redirect(url_for('dashboard'))
            try:
                plot_filename, positive_count, neutral_count, negative_count = analyze_reviews(file_path, product_id, start_time, end_time)
                if plot_filename is None:
                    flash('No reviews found for the given product ID and timeframe')
                    return redirect(url_for('dashboard'))
                new_entry = AnalysisEntry(
                    user_id=session['user_id'],
                    product_id=product_id,
                    timeframe=f"{start_time}-{end_time}",
                    plot_filename=plot_filename,
                    positive_count=positive_count,
                    neutral_count=neutral_count,
                    negative_count=negative_count
                )
                db.session.add(new_entry)
                db.session.commit()
                return render_template('dashboard.html', file_uploaded=filename, product_id=product_id, timeframe=f"{start_time}-{end_time}", plot_filename=plot_filename)
            except Exception as e:
                flash(f"Error analyzing reviews: {str(e)}")
                return redirect(url_for('dashboard'))
    user_entries = AnalysisEntry.query.filter_by(user_id=session['user_id']).all()
    return render_template('dashboard.html', entries=user_entries)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        if not os.path.exists('reviews_analysis_model.pkl'):
            train_model()
    app.run(debug=True, host='127.0.0.1')
