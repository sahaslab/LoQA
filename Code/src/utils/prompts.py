from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from jinja2 import Environment, FileSystemLoader
import os, re

def _load_prompt(prompt_file_path: str, **jinja_vars) -> str:
    "Load .txt file, render with jinja2 and return LangChain template"
    if not os.path.exists(prompt_file_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file_path}")
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Only process as Jinja2 if we detect actual Jinja2 syntax (variables or control structures)
    has_jinja2_vars = re.search(r'\{\{\s*\w+\s*\}\}', content)  # {{ variable }}
    has_jinja2_control = "{%" in content  # {% if %}, {% for %}, etc.
    
    if has_jinja2_vars or has_jinja2_control:
      prompts_dir = os.path.dirname(os.path.abspath(prompt_file_path))
      env = Environment(loader=FileSystemLoader(prompts_dir), trim_blocks=True, lstrip_blocks=True)
      template = env.from_string(content)
      content = template.render(**jinja_vars)
      # Convert {{var}} to {var} for LangChain
      content = re.sub(r'\{\{\s*(\w+)\s*\}\}', r'{\1}', content)
    return content

def question_generation_prompt_template(model, prompt_file_path: str, **jinja_vars):
    template = _load_prompt(prompt_file_path, **jinja_vars)
    prompt_template = PromptTemplate(
        input_variables = ['role', 'document', 'gt_arguments'],
        template = template
    )
    return prompt_template | model | StrOutputParser(), prompt_template

def dynamic_question_generation_prompt_template(model, prompt_file_path: str, **jinja_vars):
    template = _load_prompt(prompt_file_path, **jinja_vars)
    prompt_template = PromptTemplate(
        input_variables = ['role', 'document'],
        template = template
    )
    return prompt_template | model | StrOutputParser(), prompt_template

def argument_extraction_prompt_template(model, prompt_file_path: str, **jinja_vars):
    template = _load_prompt(prompt_file_path, **jinja_vars)
    prompt_template = PromptTemplate(
        input_variables = ['role', 'document', 'role_question'],
        template = template
    )
    return prompt_template | model | StrOutputParser(), prompt_template
    #returning prompt template for to see the prompt.
 
def vllm_argument_extraction_prompt_template(prompt_file_path: str, **jinja_vars):
    template = _load_prompt(prompt_file_path, **jinja_vars)
    prompt_template = PromptTemplate(
        input_variables = ['role', 'document', 'role_question'],
        template = template
    )
    return prompt_template

def vllm_question_generation_prompt_template(prompt_file_path: str, **jinja_vars):
    template = _load_prompt(prompt_file_path, **jinja_vars)
    prompt_template = PromptTemplate(
        input_variables = ['role', 'document', 'gt_arguments'],
        template = template
    )
    return prompt_template

def loqa_refinement_prompt_template(model, prompt_file_path: str, **jinja_vars):
    """Refinement prompt for feedback loop iterations"""
    template = _load_prompt(prompt_file_path, **jinja_vars)
    prompt_template = PromptTemplate(
        input_variables=[
            'role',
            'document', 
            'gt_arguments',
            'iteration',
            'current_questions',
            'precision',
            'recall',
            'f1_score',
            'n_matched_gt',
            'n_gt',
            'n_pred',
            'n_matched_pred',
            'matched_arguments',
            'missing_arguments',
            'extra_arguments'
        ],
        template=template
    )
    return prompt_template | model | StrOutputParser(), prompt_template

def opt_leakage_check_prompt_template(model, prompt_file_path: str, **jinja_vars):
    template = _load_prompt(prompt_file_path, **jinja_vars)
    prompt_template = PromptTemplate(
        input_variables=['role', 'current_questions', 'gt_arguments'],
        template=template
    )
    return prompt_template | model | StrOutputParser(), prompt_template