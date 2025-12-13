# ML-AI-GoTo-Docs

**Your Go-To Resource for Machine Learning & AI Concepts**

Welcome to **ML-AI-GoTo-Docs**, a collaborative project designed to provide clear, code-focused documentation and explanations for fundamental and advanced Machine Learning and Artificial Intelligence concepts. Built by a team of three, this platform aims to bridge the gap between theory and implementation.

## 🚀 About the Project

This project acts as a living documentation site where we aggregate knowledge, mathematical foundations, and practical code examples. Our goal is to create a resource where you can find:

- **Clear Explanations**: "Why is this used in ML?"
- **Code with Context**: Real-world Python implementations using standard libraries like `numpy`, `scipy`, and `pandas`.
- **Searchable Content**: Quickly find the definitions and examples you need.

## ✨ Features

- **Probability & Statistics**:
  - Comprehensive coverage of Probability Basics (Bayes Theorem, Random Variables).
  - Deep dives into Probability Distributions (Normal, Poisson, Binomial, etc.).
  - Statistical measures (Descriptive stats, Correlation, outliers).
- **Core AI Concepts**:
  - Introduction to AI landscapes.
- **Interactive Search**: Real-time server-side search to navigate topics instantly.
- **Code-First Approach**: Every concept is backed by executable Python code blocks directly in the documentation.

## 🛠️ Technology Stack

- **Backend**: [Flask](https://flask.palletsprojects.com/) (Python)
- **Data Science**: `numpy`, `pandas`, `scipy`
- **Frontend**: HTML5, CSS3, Jinja2 Templates
- **Structure**: Modular Flask Blueprint architecture

## 🏁 Getting Started

Follow these instructions to set up the project locally.

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/bonzainsights/ML-AI-GoTo-Docs.git
    cd ML-AI-GoTo-Docs
    ```

2.  **Create a virtual environment (Recommended)**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Start the Flask server**:

    ```bash
    python run.py
    ```

2.  **Access the docs**:
    Open your browser and navigate to `http://127.0.0.1:5000`.

## 🗺️ Roadmap & TODOs

We have an ambitious roadmap to cover the vast landscape of ML/AI. Here is what we are working on:

- [ ] **Machine Learning Fundamentals**:
  - Linear/Logistic Regression (Manual implementation & scikit-learn).
  - Decision Trees & Random Forests.
  - SVM & KNN.
- [ ] **Deep Learning**:
  - Neural Networks from scratch.
  - Backpropagation explained visually.
  - Framework guides (PyTorch/TensorFlow).
- [ ] **Advanced Architectures**:
  - CNNs for Computer Vision.
  - RNNs/LSTMs and Transformers for NLP.
- [ ] **Optimization**: Gradient Descent variants, Regularization techniques.
- [ ] **UI/UX**: Enhanced interactive visualizations (JS/D3.js).

## 🤝 Contributing

We are a team of three building this together!
If you are looking to contribute, please check out the [Contribution Guide](/contribute) running locally or view the `app/templates/how_to_contribute.html` file.
**Key Rule**: Always explain _why_ a concept is relevant to ML/AI and provide clear, beginner-friendly code examples.

## 📄 License

[MIT License](LICENSE) (or applicable license)
