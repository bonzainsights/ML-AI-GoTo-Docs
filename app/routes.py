from flask import Blueprint, render_template, jsonify, request, current_app
import numpy as np
from scipy import stats
import pandas as pd
import re
import os

main = Blueprint('main', __name__)

# Global Cache
SEARCH_INDEX_CACHE = None

def get_or_build_index():
    global SEARCH_INDEX_CACHE
    if SEARCH_INDEX_CACHE is not None:
        return SEARCH_INDEX_CACHE

    index = []
    
    # Define pages to crawl: filename -> (url_endpoint, base_title)
    pages = {
        'index.html': ('main.index', 'Home'),
        'ai_intro.html': ('main.ai_intro', 'AI Introduction'),
        'statistics.html': ('main.statistics_page', 'Statistics'),
        'probability_basics.html': ('main.probability_basics', 'Probability Basics'),
        'probability_distributions.html': ('main.probability_distributions', 'Probability Distributions'),
        'statistical_inference.html': ('main.statistical_inference', 'Statistical Inference'),
        'how_to_contribute.html': ('main.contribute', 'Contribution Guide')
    }
    
    template_dir = os.path.join(current_app.root_path, 'templates')
    
    for filename, (endpoint, page_title) in pages.items():
        filepath = os.path.join(template_dir, filename)
        if not os.path.exists(filepath):
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 1. Add the Page itself
        # Resolve URL (simple hack: we don't have request context for url_for sometimes, 
        # so we map manually or use url_for in a request context if possible. 
        # For simplicity, we hardcode paths or rely on endpoint names if frontend handles it.
        # Ideally, we should use url_for inside a request context context.)
        base_url = "/" if filename == 'index.html' else "/" + filename.replace('.html', '').replace('_page', '')
        # Fix specific manual overrides
        if filename == 'statistics.html': base_url = '/statistics'
        if filename == 'ai_intro.html': base_url = '/intro'
        if filename == 'probability_basics.html': base_url = '/probability-basics'
        if filename == 'probability_distributions.html': base_url = '/probability-distributions'
        if filename == 'statistical_inference.html': base_url = '/statistical-inference'
        if filename == 'how_to_contribute.html': base_url = '/contribute'

        index.append({
            'title': page_title,
            'url': base_url,
            'category': 'Page'
        })

        # 2. Extract Sections (Regex)
        # Looking for <section id="pid"> ... <h2>Title</h2>
        # This is a simple regex parser.
        section_matches = re.finditer(r'<section\s+id=["\']([^"\']+)["\'][^>]*>\s*<h[2-3][^>]*>(.*?)</h[2-3]>', content, re.DOTALL)
        
        for match in section_matches:
            sec_id = match.group(1)
            raw_title = match.group(2)
            # Clean HTML tags from title
            clean_title = re.sub(r'<[^>]+>', '', raw_title).strip()
            # Remove "X. " prefix if present (e.g. "1. Mean")
            clean_title = re.sub(r'^\d+\.\s*', '', clean_title)
            
            index.append({
                'title': f"{page_title}: {clean_title}",
                'url': f"{base_url}#{sec_id}",
                'category': 'Section'
            })
            
    SEARCH_INDEX_CACHE = index
    return index

@main.route('/api/search')
def search():
    query = request.args.get('q', '').lower().strip()
    if not query:
        return jsonify([])
        
    index = get_or_build_index()
    results = []
    
    for item in index:
        if query in item['title'].lower():
            results.append(item)
            
    # Limit results
    return jsonify(results[:10])

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/favicon.ico')
def favicon():
    return current_app.send_static_file('favicon.ico')

@main.route('/.well-known/appspecific/com.chrome.devtools.json')
def chrome_devtools():
    return jsonify({}), 200

@main.route('/intro')
def ai_intro():
    return render_template('ai_intro.html')

@main.route('/probability-basics')
def probability_basics():
    # --- 1. Probability Rules (Bayes) ---
    # P(A|B) = P(B|A) * P(A) / P(B)
    # Scenario: Disease testing
    p_disease = 0.01        # P(A)
    p_pos_given_disease = 0.99 # P(B|A) (Sensitivity)
    p_pos_given_no_disease = 0.05 # False positive rate
    
    # P(B) = P(B|A)P(A) + P(B|~A)P(~A)
    p_pos = (p_pos_given_disease * p_disease) + (p_pos_given_no_disease * (1 - p_disease))
    
    p_disease_given_pos = (p_pos_given_disease * p_disease) / p_pos
    
    bayes_data = {
        'p_a': p_disease,
        'p_b_given_a': p_pos_given_disease,
        'p_b': round(p_pos, 4),
        'result': round(p_disease_given_pos, 4)
    }

    # --- 2. Random Variables (Scipy) ---
    # Normal Distribution (Continuous)
    norm_rv = stats.norm(loc=0, scale=1)
    norm_pdf_0 = norm_rv.pdf(0)
    norm_cdf_0 = norm_rv.cdf(0) # Should be 0.5
    
    # Binomial Distribution (Discrete)
    # 10 trials, p=0.5
    binom_rv = stats.binom(n=10, p=0.5)
    binom_pmf_5 = binom_rv.pmf(5) # Prob of exactly 5 heads
    
    rv_data = {
        'norm_pdf_0': round(norm_pdf_0, 4),
        'norm_cdf_0': round(norm_cdf_0, 4),
        'binom_pmf_5': round(binom_pmf_5, 4)
    }

    # --- 3. Expectation & Variance (Numpy) ---
    # Discrete case: Rolling a fair die
    die_outcomes = np.array([1, 2, 3, 4, 5, 6])
    die_probs = np.array([1/6] * 6)
    
    expected_value = np.sum(die_outcomes * die_probs)
    # Var(X) = E[X^2] - (E[X])^2
    expected_sq = np.sum((die_outcomes ** 2) * die_probs)
    variance = expected_sq - (expected_value ** 2)
    
    ev_data = {
        'expected_value': round(expected_value, 2),
        'variance': round(variance, 2)
    }

    # --- 4. Joint & Marginal (Pandas) ---
    # Scenario: Weather (Sunny, Rainy) vs Commute (Bus, Car)
    data = {
        'Weather': ['Sunny', 'Sunny', 'Rainy', 'Sunny', 'Rainy', 'Rainy', 'Sunny', 'Rainy', 'Sunny', 'Rainy'],
        'Commute': ['Walk', 'Bus',  'Bus',   'Walk',  'Car',   'Bus',   'Walk',  'Car',   'Bus',   'Car']
    }
    df = pd.DataFrame(data)
    joint_probs = pd.crosstab(df['Weather'], df['Commute'], normalize=True)
    
    # Marginal probs
    marginal_weather = joint_probs.sum(axis=1) # Sum over columns to get row sums
    marginal_commute = joint_probs.sum(axis=0) # Sum over rows to get column sums
    
    joint_data = {
        'joint': joint_probs.to_string(),
        'marginal_weather': marginal_weather.to_dict(),
        'marginal_commute': marginal_commute.to_dict()
    }

    return render_template('probability_basics.html', 
                           bayes=bayes_data, 
                           rv=rv_data, 
                           ev=ev_data, 
                           joint=joint_data)

@main.route('/statistics')
def statistics_page():
    # Dataset 1 (X)
    data_x = [12, 15, 12, 18, 20, 22, 12, 25, 30]
    # Dataset 2 (Y) - correlated somewhat with X for demonstration
    data_y = [11, 14, 13, 19, 21, 24, 11, 26, 31]
    
    # --- Manual Calculations ---
    # Variance
    mean_val = sum(data_x) / len(data_x)
    manual_var = sum((x - mean_val) ** 2 for x in data_x) / (len(data_x) - 1)
    # Std Dev
    manual_std = manual_var ** 0.5
    # Covariance
    mean_x = sum(data_x) / len(data_x)
    mean_y = sum(data_y) / len(data_y)
    manual_cov = sum((data_x[i] - mean_x) * (data_y[i] - mean_y) for i in range(len(data_x))) / (len(data_x) - 1)
    # Range
    manual_range = max(data_x) - min(data_x)
    # Percentiles (Approximate manual method for demonstration)
    sorted_x = sorted(data_x)
    manual_p25 = sorted_x[int(0.25 * len(sorted_x))]
    manual_p75 = sorted_x[int(0.75 * len(sorted_x))]
    manual_iqr = manual_p75 - manual_p25
    
    # Correlation (Manual)
    # Need std_y first
    manual_var_y = sum((y - mean_y) ** 2 for y in data_y) / (len(data_y) - 1)
    manual_std_y = manual_var_y ** 0.5
    manual_corr = manual_cov / (manual_std * manual_std_y)

    # --- Numpy Calculations ---
    np_mean = np.mean(data_x)
    np_median = np.median(data_x)
    # Numpy Mode (Custom using unique)
    vals, counts = np.unique(data_x, return_counts=True)
    np_mode = vals[np.argmax(counts)]
    
    np_var = np.var(data_x, ddof=1) # Sample variance
    np_std = np.std(data_x, ddof=1) # Sample std dev
    np_cov = np.cov(data_x, data_y)[0][1] # Covariance matrix
    
    # Range
    np_range = np.ptp(data_x)
    # Percentiles
    np_p25 = np.percentile(data_x, 25)
    np_p75 = np.percentile(data_x, 75)
    np_iqr = np_p75 - np_p25
    # Correlation
    np_corr = np.corrcoef(data_x, data_y)[0][1]
    
    # --- Scipy Calculations ---
    # Scipy Calculations
    # Mean (tmean)
    scipy_mean = stats.tmean(data_x)
    # Median (scoreatpercentile)
    scipy_median = stats.scoreatpercentile(data_x, 50)
    # Mode
    scipy_mode_result = stats.mode(data_x, keepdims=True)
    scipy_mode = scipy_mode_result.mode[0]
    # Variance & Std Dev
    scipy_var = stats.tvar(data_x)
    scipy_std = stats.tstd(data_x)
    # Covariance (Derived from Pearson Correlation)
    # cov(x,y) = pearsonr(x,y) * std(x) * std(y)
    r_val, _ = stats.pearsonr(data_x, data_y)
    scipy_cov = r_val * stats.tstd(data_x) * stats.tstd(data_y)
    
    # IQR
    scipy_iqr = stats.iqr(data_x)
    # Percentiles
    scipy_p25 = stats.scoreatpercentile(data_x, 25)
    scipy_p75 = stats.scoreatpercentile(data_x, 75)
    # Correlation
    scipy_pearson, _ = stats.pearsonr(data_x, data_y)
    scipy_spearman, _ = stats.spearmanr(data_x, data_y)
    
    # --- Pandas Calculations ---
    df = pd.DataFrame({'x': data_x, 'y': data_y})
    pd_mean = df['x'].mean()
    pd_median = df['x'].median()
    pd_mode = df['x'].mode()[0]
    pd_var = df['x'].var()
    pd_std = df['x'].std()
    pd_cov = df['x'].cov(df['y'])
    
    # Range
    pd_range = df['x'].max() - df['x'].min()
    # Percentiles
    pd_p25 = df['x'].quantile(0.25)
    pd_p75 = df['x'].quantile(0.75)
    pd_iqr = pd_p75 - pd_p25
    # Correlation
    pd_pearson = df['x'].corr(df['y'], method='pearson')
    pd_spearman = df['x'].corr(df['y'], method='spearman')

    return render_template('statistics.html', 
                           dataset_x=data_x,
                           dataset_y=data_y,
                           stats={
                               'manual': {
                                   'var': manual_var,
                                   'std': manual_std,
                                   'cov': manual_cov,
                                   'range': manual_range,
                                   'iqr': manual_iqr,
                                   'corr': manual_corr
                               },
                               'numpy': {
                                   'mean': np_mean, 
                                   'median': np_median,
                                   'mode': np_mode,
                                   'var': np_var,
                                   'std': np_std,
                                   'cov': np_cov,
                                   'range': np_range,
                                   'iqr': np_iqr,
                                   'p25': np_p25,
                                   'p75': np_p75,
                                   'corr': np_corr
                               },
                               'scipy': {
                                   'mean': scipy_mean,
                                   'median': scipy_median,
                                   'mode': scipy_mode,
                                   'var': scipy_var,
                                   'std': scipy_std,
                                   'cov': scipy_cov,
                                   'iqr': scipy_iqr,
                                   'p25': scipy_p25,
                                   'p75': scipy_p75,
                                   'pearson': scipy_pearson,
                                   'spearman': scipy_spearman
                               },
                               'pandas': {
                                   'mean': pd_mean, 
                                   'median': pd_median, 
                                   'mode': pd_mode,
                                   'var': pd_var,
                                   'std': pd_std,
                                   'cov': pd_cov,
                                   'range': pd_range,
                                   'iqr': pd_iqr,
                                   'p25': pd_p25,
                                   'p75': pd_p75,
                                   'pearson': pd_pearson,
                                   'spearman': pd_spearman
                               }
                           })
@main.route('/probability-distributions')
def probability_distributions():
    # --- 1. Discrete ---
    # Bernoulli (p=0.6)
    bernoulli_val = stats.bernoulli.pmf(1, p=0.6)
    
    # Binomial (n=10, p=0.5, k=5)
    binom_val = stats.binom.pmf(5, n=10, p=0.5)
    
    # Geometric (p=0.2, k=3)
    geom_val = stats.geom.pmf(3, p=0.2)
    
    # Poisson (mu=3, k=5)
    poisson_val = stats.poisson.pmf(5, mu=3)
    
    discrete_data = {
        'bernoulli': round(bernoulli_val, 4),
        'binomial': round(binom_val, 4),
        'geometric': round(geom_val, 4),
        'poisson': round(poisson_val, 4)
    }

    # --- 2. Continuous ---
    # Uniform (0, 10, x=5)
    uniform_val = stats.uniform.pdf(5, loc=0, scale=10)
    
    # Normal (0, 1, x=0)
    normal_val = stats.norm.pdf(0, loc=0, scale=1)
    
    # Exponential (scale=2, x=2). note: scale=1/lambda. if lambda=0.5, scale=2.
    expon_val = stats.expon.pdf(2, scale=2)
    
    # Gamma (a=2, scale=2, x=3)
    gamma_val = stats.gamma.pdf(3, a=2, scale=2)
    
    # Beta (a=2, b=2, x=0.5)
    beta_val = stats.beta.pdf(0.5, a=2, b=2)
    
    # Chi-square (df=2, x=1)
    chisquare_val = stats.chi2.pdf(1, df=2)
    
    continuous_data = {
        'uniform': round(uniform_val, 4),
        'normal': round(normal_val, 4),
        'exponential': round(expon_val, 4),
        'gamma': round(gamma_val, 4),
        'beta': round(beta_val, 4),
        'chisquare': round(chisquare_val, 4)
    }

    return render_template('probability_distributions.html',
                           discrete=discrete_data,
                           continuous=continuous_data)

@main.route('/contribute')
def contribute():
    return render_template('how_to_contribute.html')

@main.route('/statistical-inference')
def statistical_inference():
    # --- 1. Sampling Methods ---
    # Population of 1000 items
    population = np.arange(1000)
    # Simple Random Sampling (n=50)
    simple_sample = np.random.choice(population, size=50, replace=False)
    simple_sample_mean = np.mean(simple_sample)
    
    # --- 2. Law of Large Numbers (LLN) ---
    # Simulate coin flips (0=Tails, 1=Heads). True mean = 0.5
    # Small N
    small_n = 10
    small_flips = np.random.binomial(n=1, p=0.5, size=small_n)
    small_mean = np.mean(small_flips)
    # Large N
    large_n = 10000
    large_flips = np.random.binomial(n=1, p=0.5, size=large_n)
    large_mean = np.mean(large_flips)
    
    # --- 3. Central Limit Theorem (CLT) ---
    # Population: Uniform [0, 100]. Mean = 50.
    # Take 1000 samples of size 30, calc means.
    sample_means = [np.mean(np.random.uniform(0, 100, 30)) for _ in range(1000)]
    clt_mean = np.mean(sample_means)
    clt_std = np.std(sample_means) # Should be close to (100-0)/sqrt(12) / sqrt(30) ≈ 28.8 / 5.47 ≈ 5.26
    
    # --- 4. Estimators ---
    # Bias & Variance Demonstration is theoretical in text, but let's show MLE fit.
    # Data from Normal(5, 2)
    sample_data = np.random.normal(loc=5, scale=2, size=100)
    # MLE estimation of parameters
    mu_mle, std_mle = stats.norm.fit(sample_data)
    
    # --- 5. Confidence Intervals ---
    # 95% CI for the mean of sample_data
    # stats.sem = standard error of mean = std / sqrt(n)
    ci_low, ci_high = stats.norm.interval(0.95, loc=np.mean(sample_data), scale=stats.sem(sample_data))
    
    # --- 6. Hypothesis Testing ---
    # A. One-sample t-test
    # H0: Mean = 5.0 vs H1: Mean != 5.0
    t_stat, t_p_val = stats.ttest_1samp(sample_data, 5.0)
    
    # B. Chi-square Goodness of Fit
    # Observed counts of rolling a die 60 times
    observed = np.array([12, 8, 11, 9, 13, 7])
    # Expected counts (fair die) = 10 each
    expected = np.array([10, 10, 10, 10, 10, 10])
    chi2_stat, chi2_p_val = stats.chisquare(f_obs=observed, f_exp=expected)
    
    results = {
        'sampling': {'mean': round(simple_sample_mean, 2)},
        'lln': {'small_mean': small_mean, 'large_mean': round(large_mean, 4)},
        'clt': {'mean': round(clt_mean, 2), 'std': round(clt_std, 2)},
        'estimators': {'mu_mle': round(mu_mle, 2), 'std_mle': round(std_mle, 2)},
        'ci': {'low': round(ci_low, 2), 'high': round(ci_high, 2)},
        'tests': {
            't_stat': round(t_stat, 2), 
            't_p': round(t_p_val, 4),
            'chi2_stat': round(chi2_stat, 2),
            'chi2_p': round(chi2_p_val, 4)
        }
    }
    
    return render_template('statistical_inference.html', res=results)
