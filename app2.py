import streamlit as st
import numpy as np
import pandas as pd
import sqlite3
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

# ================================
# PART 1: Database Setup
# ================================
foods = ["Idly", "Dosa", "Vada", "Roti", "Meals", "Veg Biryani",
         "Egg Biryani", "Chicken Biryani", "Mutton Biryani",
         "Ice Cream", "Noodles", "Manchooriya", "Orange juice",
         "Apple Juice", "Pineapple juice", "Banana juice"]

def init_data():
    conn = sqlite3.connect('Restaurant_food_data.db')
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS item (
                    item_name TEXT,
                    no_of_customers TEXT,
                    no_of_positives TEXT,
                    no_of_negatives TEXT,
                    pos_perc TEXT,
                    neg_perc TEXT
                )""")
    c.execute("SELECT * FROM item")
    existing_items = c.fetchall()
    for food in foods:
        if not any(food in item for item in existing_items):
            c.execute("INSERT INTO item VALUES (?,?,?,?,?,?)",
                      (food, "0", "0", "0", "0.0%", "0.0%"))
    conn.commit()
    conn.close()

init_data()

# ================================
# PART 2: Data Training
# ================================
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
corpus = []
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in set(all_stopwords)]
    corpus.append(' '.join(review))

# Use TF-IDF instead of CountVectorizer
tfidf = TfidfVectorizer(max_features=2000)
X = tfidf.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# ================================
# PART 2.1: Model Selector
# ================================
@st.cache_resource
def train_model(model_choice):
    if model_choice == "Naive Bayes":
        model = GaussianNB()
    elif model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=200, random_state=0)
    elif model_choice == "SVM":
        model = LinearSVC(random_state=0)
    else:
        model = GaussianNB()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rep = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
    cm = confusion_matrix(y_test, y_pred)
    return model, acc, rep, cm

# Default Model
model_choice = st.sidebar.selectbox("🔍 Choose ML Model", ["Naive Bayes", "Random Forest", "SVM"])
classifier, accuracy, report, cm = train_model(model_choice)

# ================================
# PART 3: Helper Functions
# ================================
def estimate(review_text, selected_foods):
    conn = sqlite3.connect('Restaurant_food_data.db')
    c = conn.cursor()

    review = re.sub('[^a-zA-Z]', ' ', review_text)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in set(all_stopwords)]
    review = ' '.join(review)

    X = tfidf.transform([review]).toarray()
    res = classifier.predict(X)

    if "not" in review:
        res[0] = abs(res[0] - 1)

    c.execute("SELECT *, oid FROM item")
    records = c.fetchall()

    for rec in records:
        rec = list(rec)
        if rec[0] in selected_foods:
            n_cust = int(rec[1]) + 1
            n_pos = int(rec[2])
            n_neg = int(rec[3])
            if res[0] == 1:
                n_pos += 1
            else:
                n_neg += 1
            pos_percent = round((n_pos / n_cust) * 100, 1)
            neg_percent = round((n_neg / n_cust) * 100, 1)
            c.execute("""UPDATE item SET 
                            no_of_customers=?, 
                            no_of_positives=?, 
                            no_of_negatives=?, 
                            pos_perc=?, 
                            neg_perc=? WHERE oid=?""",
                      (str(n_cust), str(n_pos), str(n_neg),
                       f"{pos_percent}%", f"{neg_percent}%", str(rec[-1])))
    conn.commit()
    conn.close()


def get_data():
    conn = sqlite3.connect('Restaurant_food_data.db')
    df = pd.read_sql_query("SELECT * FROM item", conn)
    conn.close()
    return df


def clear_selected(items):
    conn = sqlite3.connect('Restaurant_food_data.db')
    c = conn.cursor()
    for food in items:
        c.execute("""UPDATE item 
                     SET no_of_customers=?, no_of_positives=?, no_of_negatives=?, 
                         pos_perc=?, neg_perc=? 
                     WHERE item_name=?""",
                  ("0", "0", "0", "0.0%", "0.0%", food))
    conn.commit()
    conn.close()


def clear_all():
    clear_selected(foods)

# ================================
# PART 4: Streamlit App
# ================================
st.title("🍴 Restaurant Review Analysis System")
st.write("An ML-powered system to analyze restaurant reviews with **Naive Bayes / Random Forest / SVM**.")

menu = st.sidebar.radio("Navigation", ["Home", "Customer", "Owner", "Analytics"])

if menu == "Home":
    st.subheader("Welcome")
    st.write("Choose whether you are a **Customer** or the **Owner** from the sidebar.")

elif menu == "Customer":
    st.subheader("Leave a Review")
    st.write("Select the food items you had:")

    # Initialize session state for checkboxes
    if 'selected_foods' not in st.session_state:
        st.session_state.selected_foods = {food: False for food in foods}

    # Display all food items as checkboxes
    selected_foods = []
    for food in foods:
        st.session_state.selected_foods[food] = st.checkbox(food, value=st.session_state.selected_foods[food])
        if st.session_state.selected_foods[food]:
            selected_foods.append(food)

    review_text = st.text_area("Write your review:")

    if st.button("Submit Review"):
        if review_text and selected_foods:
            estimate(review_text, selected_foods)
            st.success("✅ Review submitted successfully!")

            # Reset checkboxes
            for food in foods:
                st.session_state.selected_foods[food] = False
        else:
            st.error("⚠️ Please select food items and write a review.")



elif menu == "Owner":
    st.subheader("Owner Login")
    rras_code = "sarth"
    code_input = st.text_input("Enter Owner Code:", type="password")
    if st.button("Verify"):
        if code_input == rras_code:
            st.success("Access Granted ✅")

            st.subheader("📊 Current Database")
            df = get_data()
            st.dataframe(df)

            st.subheader("🧹 Clear Data")
            clear_option = st.radio("Choose:", ["Clear Selected", "Clear All"])
            if clear_option == "Clear Selected":
                items_to_clear = st.multiselect("Select items to clear:", foods)
                if st.button("Clear Selected Data"):
                    clear_selected(items_to_clear)
                    st.success("Selected items cleared.")
            elif clear_option == "Clear All":
                if st.button("Clear All Data"):
                    clear_all()
                    st.success("All items cleared.")
        else:
            st.error("❌ Incorrect Code")

elif menu == "Analytics":
    # ==============================
    # 1️⃣ Reviews per Item Chart
    # ==============================
    st.subheader("📊 Reviews per Item")
    df = get_data()
    review_counts = df[['item_name', 'no_of_positives', 'no_of_negatives']].copy()
    review_counts['no_of_positives'] = review_counts['no_of_positives'].astype(int)
    review_counts['no_of_negatives'] = review_counts['no_of_negatives'].astype(int)

    fig1, ax1 = plt.subplots(figsize=(12,6))
    width = 0.35
    x = np.arange(len(review_counts['item_name']))

    ax1.bar(x - width/2, review_counts['no_of_positives'], width, label='Positive Reviews', color='green')
    ax1.bar(x + width/2, review_counts['no_of_negatives'], width, label='Negative Reviews', color='red')

    ax1.set_xticks(x)
    ax1.set_xticklabels(review_counts['item_name'], rotation=45, ha='right')
    ax1.set_ylabel("Number of Reviews")
    ax1.set_title("Positive vs Negative Reviews per Item")
    ax1.legend()
    ax1.grid(axis='y')

    st.pyplot(fig1)

    # ==============================
    # 2️⃣ Database Snapshot
    # ==============================
    st.subheader("Database Snapshot")
    st.dataframe(df)

    # ==============================
    # 3️⃣ Model Performance
    # ==============================
    st.subheader("📈 Model Performance")
    st.write(f"**Selected Model:** {model_choice}")
    st.write(f"**Accuracy:** {accuracy*100:.2f}%")
    st.text("Classification Report:")
    st.text(report)

    st.subheader("Confusion Matrix")
    fig2, ax2 = plt.subplots(figsize=(5,4))  # Smaller size
    im = ax2.imshow(cm, cmap=plt.cm.Blues)
    ax2.set_title("Confusion Matrix")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    classes = ['Negative', 'Positive']
    ax2.set_xticks(np.arange(len(classes)))
    ax2.set_yticks(np.arange(len(classes)))
    ax2.set_xticklabels(classes)
    ax2.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax2.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black")
    st.pyplot(fig2)

