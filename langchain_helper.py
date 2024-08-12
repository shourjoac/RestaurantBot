from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from secret_key import api_key

llm = GooglePalm(google_api_key=api_key, temperature= 0.5)

def generate_restaurant_name_and_items(cuisine):
    prompt_template_name = PromptTemplate(
        input_variables=['cuisine'],
        template="I want to open a restaurant for {cuisine} food. Suggest me one fancy and innovative name. Do not give more than one name"
    )
    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")
    prompt_template_name_2 = PromptTemplate(
        input_variables=['restaurant_name'],
        template="Please suggest menu food items for {restaurant_name} restaurant. Comma separated list of food items to be displayed"
    )
    item_chain = LLMChain(llm=llm, prompt=prompt_template_name_2, output_key="menu_items")
    chain = SequentialChain(
        chains=[name_chain, item_chain],
        input_variables=['cuisine'],
        output_variables=['restaurant_name', 'menu_items']
    )
    response = chain({'cuisine': cuisine})
    return response

if __name__ == "__main__":
    print(generate_restaurant_name_and_items("Italian"))