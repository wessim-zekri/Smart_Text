from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import re
 
def createModel():
    llm = ChatGroq(
            model='llama-3.2-90b-text-preview',
            api_key='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',
            temperature=0,
            verbose=True,
        )
    return llm
''' 
def extract_code_block(text):
    #extract code block from generated code
    match = re.search(r"```(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None
'''
def generateTrans(llm,text,trg_lang, src_lang):
    template = """
    <|start_header_id|>system<|end_header_id|>
    You are a reliable translation assistant. Your task is to translate or rewrite the text precisely.
    
    Instructions:
    - Translate the given text: "{text}" from {src_lang} to {trg_lang}.
    - Only return the translation or, if it’s already in the target language, the original word as is.
    - Do not add any comments, explanations, or questions.
    - If the word cannot be translated due to ambiguity or lack of context, simply rewrite it without any additional information.
    
    Output only the translated or original text directly.
    <|eot_id|>
"""
    prompt = PromptTemplate.from_template(template)
    prompt_text = prompt.format(text=text,trg_lang=trg_lang, src_lang=src_lang)
    chain2 = llm | StrOutputParser()
    result = chain2.invoke(prompt_text)
    #result =extract_code_block(result)
    return result

#llm = createModel()
#text = ''
#lang = ''

#res = generateTrans(llm,text,trg_lang, src_lang)

#print(res)
