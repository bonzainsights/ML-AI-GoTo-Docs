from flask import Blueprint, render_template, jsonify, request, current_app
import calc.calculate_static_values as calc
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
        'regression.html': ('main.regression_page', 'Regression & Linear Models'),
        'bayesian_statistics.html': ('main.bayesian_statistics', 'Bayesian Statistics'),
        'multivariate_statistics.html': ('main.multivariate_statistics', 'Multivariate Statistics'),
        'statistical_learning.html': ('main.statistical_learning', 'Statistical Learning Concepts'),
        'experimental_design.html': ('main.experimental_design', 'Experimental Design & Evaluation'),
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
        if filename == 'regression.html': base_url = '/regression'
        if filename == 'bayesian_statistics.html': base_url = '/bayesian-statistics'
        if filename == 'multivariate_statistics.html': base_url = '/multivariate-statistics'
        if filename == 'statistical_learning.html': base_url = '/statistical-learning'
        if filename == 'experimental_design.html': base_url = '/experimental-design'
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
    res = calc.probability_basics_results()
    return render_template('probability_basics.html', **res)

@main.route('/statistics')
def statistics_page():
    res = calc.statistics_results()
    return render_template('statistics.html', **res)
@main.route('/probability-distributions')
def probability_distributions():
    res = calc.probability_distributions_results()
    return render_template('probability_distributions.html', **res)

@main.route('/contribute')
def contribute():
    return render_template('how_to_contribute.html')

@main.route('/statistical-inference')
def statistical_inference():
    results = calc.statistical_inference_results()
    return render_template('statistical_inference.html', res=results)


@main.route('/regression')
def regression_page():
    results = calc.regression_results()
    return render_template('regression.html', res=results)


@main.route('/bayesian-statistics')
def bayesian_statistics():
    results = calc.bayesian_results()
    return render_template('bayesian_statistics.html', res=results)


@main.route('/multivariate-statistics')
def multivariate_statistics():
    results = calc.multivariate_results()
    return render_template('multivariate_statistics.html', res=results)


@main.route('/statistical-learning')
def statistical_learning():
    results = calc.statistical_learning_results()
    return render_template('statistical_learning.html', res=results)


@main.route('/experimental-design')
def experimental_design():
    results = calc.experimental_results()
    return render_template('experimental_design.html', res=results)


