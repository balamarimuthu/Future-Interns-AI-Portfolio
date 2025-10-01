# 🌟 Future Interns AI & Data Science Project Portfolio

Welcome to my **Future Interns program** project portfolio.  
This repository contains the complete work for **three distinct, professional-grade AI and Data Science projects**, each demonstrating a different core competency in the field — from **predictive analytics** to **generative AI**.

These projects showcase a **complete, end-to-end understanding** of the data science lifecycle, including complex data engineering, strategic model development, sophisticated system architecture, and final application deployment.

---

## 📂 Repository Structure

```text
FUTURE_INTERN_AI_PORTFOLIO/
├── 📁 01_Sales_Forecasting_System/
│   ├── 📄 project_1.ipynb
│   └── 📄 Sales Forecasting System.pdf
│
├── 📁 02_Churn_Prediction_App/
│   ├── 📄 app.py
│   ├── 📄 churn_model.pkl
│   ├── 📄 churn_prediction_system.ipynb
│   └── 📄 customer churn prediction.pdf
│
└── 📁 03_Advanced_AI_Chatbot/
    ├── 📄 chatbot_customer_support.pdf
    ├── 📄 app.py
    ├── 📄 customer_support_chatbot.ipynb (Research file)
    └── 📄 requirements.txt
```

## 📈 Project 1: Sales Forecasting System

**🔹 Task:** Build a dashboard that predicts future sales trends using historical data.

This project involved building a **high-accuracy, automated sales forecasting system** to predict future daily sales, enabling better strategic planning for inventory, marketing, and staffing.

### ✅ Key Features & Accomplishments
- **Data Engineering:** Cleaned and processed a raw transactional dataset, expertly handling inconsistent date formats and aggregating the data into a clean, univariate time series ready for modeling.  
- **Advanced Modeling:** Utilized Meta’s **Prophet** library for its superior ability to handle multiple seasonalities (weekly and yearly) and complex holiday effects automatically.  
- **Rigorous Optimization:** Improved model accuracy from a strong baseline (MAPE 4.44%) to a state-of-the-art **MAPE of 4.16%** using Prophet’s cross-validation tools for hyperparameter tuning.  
- **Actionable Insights:** Successfully decomposed sales data to identify a strong upward trend and clear weekly/yearly seasonal patterns, providing critical business insights.

### 🛠 Tools Used
- Python, Pandas, Matplotlib  
- Prophet by Meta  

---

## 👋 Project 2: Customer Churn Prediction System

**🔹 Task:** Build a machine learning model to predict which customers are likely to leave a service and present the findings in an interactive dashboard.

This project involved building a **complete, end-to-end churn prediction system**, from cleaning a messy, real-world dataset to deploying a live, interactive web application for business users.

### ✅ Key Features & Accomplishments
- **Advanced Data Cleaning:** Engineered a challenging dataset with hidden null values, inconsistent data types, and mixed categorical formats into a perfectly clean, model-ready state.  
- **Strategic Model Selection:** Conducted a professional “model bake-off,” comparing Logistic Regression, RandomForest, and XGBoost. Selected Logistic Regression as the champion based on **Recall (77%)**, the most critical business metric.  
- **Fine-Tuning for Performance:** Used RandomizedSearchCV to maximize F1-score, improving Recall to **79%**.  
- **Full Application Development:** Designed and built a **two-page Streamlit dashboard** providing both live prediction for individual customers and rich analytics visualizations.

### 🛠 Tools Used
- Python, Pandas, Scikit-learn, XGBoost,RandomForest, Logistic Regression (pickle model -Logistic)
- Streamlit (for the web application)  

---

## 🤖 Project 3: Advanced AI Support Chatbot

**🔹 Task:** Build a smart chatbot to handle real-time customer support queries using AI tools.

This was the most advanced project, involving the architecture of a **state-of-the-art, hybrid AI system** capable of handling a massive, unstructured dataset of over **2.8 million** real customer support conversations.

### ✅ Key Features & Accomplishments
- **Large-Scale Data Engineering:** Processed a **515 MB** dataset to create a clean, 100,000-conversation knowledge base using a local multi-core pipeline.  
- **Semantic Search & RAG System:** Built a complete **Retrieval-Augmented Generation (RAG)** system:
  - Created high-quality vector embeddings for all questions using **sentence-transformers**.
  - Built a highly efficient, searchable vector index with **FAISS** from Meta.  
  - This core technology prevents AI “hallucinations” by ensuring contextually relevant retrieval.  
- **Advanced Hybrid AI Architecture:** Designed and implemented a “**Smart Waiter**” system to orchestrate:
  - **Dialogflow** for simple intents.  
  - **Google Gemini API** for complex questions.  
  - All integrated within a main **Streamlit** application.  
- **Full Business Integration:** Automated creation of support tickets in **Airtable** when escalation to human support is required.

### 🛠 Tools Used
- Python, Pandas, sentence-transformers, FAISS-CPU  
- Google Dialogflow, Google Gemini API, Airtable API  
- Streamlit (final user interface & orchestrator)  
- Flask (initial webhook development phase)  

---

## 📝 How to Navigate
Each subfolder contains:
- **Code files** (`.py`)  
- **Trained models or data assets** (where applicable)  
- **Detailed project report** in Markdown  
  **Research files** (`.ipynb`)
Simply open the folder for the project you’re interested in to see all materials.

---

## 🚀 Highlights
- Demonstrated **end-to-end expertise**: data engineering → modeling → optimization → deployment.  
- Covered **predictive analytics, machine learning, and hybrid AI systems**.  
- Built fully functional **dashboards & applications** ready for business use.  

---

## 📬 Contact
For questions, feedback, or collaboration opportunities, feel free to reach out via [LinkedIn](www.linkedin.com/in/r-bala-marimuthu-380501304) or open an issue in this repository.
